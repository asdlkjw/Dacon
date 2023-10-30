from typing import Tuple, List, Dict, Optional
import torch
from torch import Tensor
from collections import OrderedDict
from torchvision.models.detection.roi_heads import (
    fastrcnn_loss,
    maskrcnn_loss,
    maskrcnn_inference,
)
from torchvision.models.detection.rpn import concat_box_prediction_layers


def has_mask(model):
    if model.roi_heads.mask_roi_pool is None:
        return False
    if model.roi_heads.mask_head is None:
        return False
    if model.roi_heads.mask_predictor is None:
        return False

    return True


def eval_forward(
    model, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]]
) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    model.eval()

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    #####proposals, proposal_losses = model.rpn(images, features, targets)
    #### rpn_loss 구하기 시작
    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [
        s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors
    ]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(
        objectness, pred_bbox_deltas
    )
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(
        proposals, objectness, images.image_sizes, num_anchors_per_level
    )

    proposal_losses = {}

    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }
    #### rpn_loss 구하기 끝

    #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    #### box_loss 구하기 시작
    image_shapes = images.image_sizes
    (
        proposals,
        matched_idxs,
        labels,
        regression_targets,
    ) = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []

    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(
        class_logits, box_regression, labels, regression_targets
    )
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

    boxes, scores, labels = model.roi_heads.postprocess_detections(
        class_logits, box_regression, proposals, image_shapes
    )
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )

    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    #### box_loss 구하기 끝

    #### mask_loss 구하기 시작
    if has_mask(model):
        mask_proposals = [p["boxes"] for p in result]

        num_images = len(proposals)
        mask_proposals = []
        pos_matched_idxs = []
        for img_id in range(num_images):
            pos = torch.where(labels[img_id] > 0)[0]
            mask_proposals.append(proposals[img_id][pos])
            pos_matched_idxs.append(matched_idxs[img_id][pos])

        if model.roi_heads.mask_roi_pool is not None:
            mask_features = model.roi_heads.mask_roi_pool(
                features, mask_proposals, image_shapes
            )
            mask_features = model.roi_heads.mask_head(mask_features)
            mask_logits = model.roi_heads.mask_predictor(mask_features)
        else:
            raise Exception("Expected mask_roi_pool to be not None")

        loss_mask = {}

        gt_masks = [t["masks"] for t in targets]
        gt_labels = [t["labels"] for t in targets]
        rcnn_loss_mask = maskrcnn_loss(
            mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
        )
        loss_mask = {"loss_mask": rcnn_loss_mask}

        labels = [r["labels"] for r in result]
        masks_probs = maskrcnn_inference(mask_logits, labels)
        for mask_prob, r in zip(masks_probs, result):
            r["masks"] = mask_prob

        losses.update(loss_mask)
    #### mask_loss 구하기 끝

    detections = result
    detections = model.transform.postprocess(
        detections, images.image_sizes, original_image_sizes
    )

    return losses, detections


import math
from typing import Any
from torchvision.models.detection.roi_heads import paste_masks_in_image
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import resize_boxes, resize_keypoints
from torch import nn


class CustomRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it performs are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(
        self,
        size_divisible: int = 32,
        fixed_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.size_divisible = size_divisible
        self.fixed_size = fixed_size
        # self._skip_resize = kwargs.pop("_skip_resize", False)

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(
                    f"images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}"
                )
            # image = self.normalize(image)
            # image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)

        return image_list, targets

    def batch_images(self, images: List[Tensor], size_divisible: int = 32) -> Tensor:
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(
            zip(result, image_shapes, original_image_sizes)
        ):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result

    def __repr__(self) -> str:
        format_string = f"{self.__class__.__name__}("
        _indent = "\n    "
        # format_string += (
        #     f"{_indent}Normalize(mean={self.image_mean}, std={self.image_std})"
        # )
        # format_string += f"{_indent}Resize(min_size={self.min_size}, max_size={self.max_size}, mode='bilinear')"
        format_string += "\n)"
        return format_string
