import random
import numpy as np
import albumentations as A


class CustomBBoxSafeRandomCrop(A.BBoxSafeRandomCrop):
    """Crop a random part of the input without loss of bboxes.
    Args:
        erosion_rate (float): erosion rate applied on input image height before crop.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes
    Image types:
        uint8, float32
    """

    def __init__(
        self, crop_height, crop_width, erosion_rate=0.0, always_apply=False, p=1.0
    ):
        super(CustomBBoxSafeRandomCrop, self).__init__(erosion_rate, always_apply, p)
        self.crop_height = crop_height
        self.crop_width = crop_width

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        if (
            len(params["bboxes"]) == 0
        ):  # less likely, this class is for use with bboxes.
            erosive_h = int(img_h * (1.0 - self.erosion_rate))
            crop_height = (
                img_h if erosive_h >= img_h else random.randint(erosive_h, img_h)
            )
            return {
                "h_start": random.random(),
                "w_start": random.random(),
                "crop_height": crop_height,
                "crop_width": int(crop_height * img_w / img_h),
            }
        # get union of all bboxes
        x, y, x2, y2 = union_of_bboxes(
            width=img_w,
            height=img_h,
            bboxes=params["bboxes"],
            erosion_rate=self.erosion_rate,
        )
        # find bigger region
        ch, cw = self.crop_height / img_h, self.crop_width / img_w

        bx, by = x, y
        # bx2, by2 = x2 + (1 - x2), y2 + (1 - y2)
        blob_w, blow_h = x2 - x, y2 - y

        by = by - (random.random() * ((ch) - blow_h))
        bx = bx - (random.random() * ((cw) - blob_w))

        crop_height = img_h if ch >= 1.0 else int(img_h * ch)
        crop_width = img_w if cw >= 1.0 else int(img_w * cw)

        h_start = np.clip(0.0 if ch >= 1.0 else by / (1.0 - ch), 0.0, 1.0)
        w_start = np.clip(0.0 if cw >= 1.0 else bx / (1.0 - cw), 0.0, 1.0)

        return {
            "h_start": h_start,
            "w_start": w_start,
            "crop_height": crop_height,
            "crop_width": crop_width,
        }


def overlap(rect1, rect2):
    """
    두 개의 사각형이 겹쳐지는지 확인하는 함수
    :param rect1: 첫번째 사각형
    :param rect2: 두번째 사각형
    :return: overlap이 되면 True, 아니면 False
    """
    return not (
        rect1[2] < rect2[0]
        or rect1[0] > rect2[2]
        or rect1[1] > rect2[3]
        or rect1[3] < rect2[1]
    )


def union_of_bboxes(height: int, width: int, bboxes, erosion_rate: float = 0.0):
    """Calculate union of bounding boxes.

    Args:
        height (float): Height of image or space.
        width (float): Width of image or space.
        bboxes (List[tuple]): List like bounding boxes. Format is `[(x_min, y_min, x_max, y_max)]`.
        erosion_rate (float): How much each bounding box can be shrinked, useful for erosive cropping.
            Set this in range [0, 1]. 0 will not be erosive at all, 1.0 can make any bbox to lose its volume.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x1, y1 = width, height
    x2, y2 = 0, 0
    bbox = random.choice(bboxes)

    x_min, y_min, x_max, y_max = bbox[:4]
    w, h = x_max - x_min, y_max - y_min
    lim_x1, lim_y1 = x_min + erosion_rate * w, y_min + erosion_rate * h
    lim_x2, lim_y2 = x_max - erosion_rate * w, y_max - erosion_rate * h
    x1, y1 = np.min([x1, lim_x1]), np.min([y1, lim_y1])
    x2, y2 = np.max([x2, lim_x2]), np.max([y2, lim_y2])
    return x1, y1, x2, y2
