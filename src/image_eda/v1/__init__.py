from typing import List, Tuple
import numpy as np
import cv2
from multiprocessing import Pool
from torch import Tensor
from PIL import Image, ImageFont, ImageDraw
import random


def divide_list_evenly(data_list: List, num_parts: int):
    if len(data_list) < num_parts:
        raise ValueError(
            f"num_parts는 len(data_list) 보다 작거나 같아야 합니다, num_parts: {num_parts}, len(data_list): {len(data_list)}"
        )

    if type(num_parts) != int:
        raise ValueError(f"num_parts는 int type이어야 합니다, num_parts: {num_parts}")

    if num_parts < 1:
        raise ValueError(f"num_parts는 0보다 커야합니다, num_parts: {num_parts}")

    total_items = len(data_list)
    items_per_part = total_items // num_parts
    remaining_items = total_items % num_parts

    divided_list = []
    start_idx = 0

    for i in range(num_parts):
        part_size = items_per_part + (1 if i < remaining_items else 0)
        end_idx = start_idx + part_size
        divided_list.append(data_list[start_idx:end_idx])
        start_idx = end_idx

    return divided_list


def calc_img_mean_std_multiprocess(
    img_path_list: List[str], multiprocess_core: int
) -> Tuple[Tuple, Tuple]:
    if type(multiprocess_core) != int:
        raise ValueError(
            f"multiprocess_core는 int type이어야 합니다, multiprocess_core: {multiprocess_core}"
        )

    if multiprocess_core < 1:
        raise ValueError("1개 이상의 작업 코어 개수를 정해주세요")

    list_chunked = divide_list_evenly(img_path_list, multiprocess_core)

    with Pool(processes=multiprocess_core) as p:
        result_list = p.starmap(calc_img_mean_std, zip(list_chunked))

    total_mean = []
    total_std = []
    for i, (mean, std) in enumerate(result_list):
        total_mean.append(mean * len(list_chunked[i]))
        total_std.append(std * len(list_chunked[i]))

    return tuple(sum(total_mean) / len(img_path_list)), tuple(
        sum(total_std) / len(img_path_list)
    )


def calc_img_mean_std(img_path_list: List[str]) -> Tuple:
    """
    input으로 넣은 이미지 데이터들의 히스토그램 mean, std 값을 구해줍니다

    Args:
        img_path_list: mean, std를 구할 이미지들의 path 리스트

    Returns:
        mean, std 값이 쌍으로 return 됩니다
    """
    img_norm = []
    img_std = []

    for path in img_path_list:
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        mean, std = np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))
        img_norm.append(mean)
        img_std.append(std)

    return np.mean(img_norm, axis=0), np.mean(img_std, axis=0)


def tensor2im(tensor: Tensor, is_zero_center: bool = True) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    if np.max(tensor) <= 1:
        if is_zero_center:
            tensor = ((tensor * 0.5) + 0.5) * 255
        else:
            tensor = tensor * 255
    img = np.transpose(tensor, (1, 2, 0)).astype(np.uint8)
    return img


def put_text(image, text, pos, color):
    img_pillow = Image.fromarray(image)
    fontpath = "gulim.ttc"
    font = ImageFont.truetype(fontpath, 24)
    draw = ImageDraw.Draw(img_pillow, "RGB")
    # color = np.append(color, 255)
    color = tuple(np.round(color).astype(np.uint8))
    draw.text(pos, text, font=font, fill=color)

    image = np.array(img_pillow)
    return image


def apply_mask(
    image, mask, labels, boxes, class_names, scores=None, mask_threshold=0.5
):
    alpha = 1
    beta = 1  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum
    COLORS = np.random.uniform(0, 255, size=(len(class_names) + 1, 3))

    if len(mask.shape) == 3:
        _, w, h = mask.shape
    else:
        _, _, w, h = mask.shape
    segmentation_map = np.zeros((w, h, 3), np.uint8)

    for n in range(len(boxes)):
        # print(n)
        if labels[n] == 0:
            continue
        else:
            color = COLORS[random.randrange(0, len(COLORS))]
            segmentation_map[:, :, 0] = np.where(
                mask[n] > mask_threshold, COLORS[labels[n]][0], 0
            )
            segmentation_map[:, :, 1] = np.where(
                mask[n] > mask_threshold, COLORS[labels[n]][1], 0
            )
            segmentation_map[:, :, 2] = np.where(
                mask[n] > mask_threshold, COLORS[labels[n]][2], 0
            )
            image = cv2.addWeighted(
                image, alpha, segmentation_map, beta, gamma, dtype=cv2.CV_8U
            )

        cv2.rectangle(
            image,
            (boxes[n][0], boxes[n][1]),
            (boxes[n][2], boxes[n][3]),
            color=color,
            thickness=2,
        )

        # put the label text above the objects
        if scores is None:
            image = put_text(
                image,
                f"{class_names[labels[n]-1]}",
                (boxes[n][0], boxes[n][1] - 30),
                color,
            )
        else:
            image = put_text(
                image,
                f"{class_names[labels[n]-1]}({scores[n]:.2f})",
                (boxes[n][0], boxes[n][1] - 30),
                color,
            )

    return image


def apply_bbox(image, labels, boxes, class_names, scores=None, color=None):
    COLORS = np.random.uniform(0, 255, size=(len(class_names) + 1, 3))

    for n in range(len(boxes)):
        if labels[n] == 0:
            continue
        else:
            if color is None:
                color = COLORS[random.randrange(0, len(COLORS))]

        cv2.rectangle(
            image,
            (boxes[n][0], boxes[n][1]),
            (boxes[n][2], boxes[n][3]),
            color=color,
            thickness=2,
        )

        # put the label text above the objects
        if scores is None:
            image = put_text(
                image,
                f"{class_names[labels[n]-1]}",
                (boxes[n][0], boxes[n][1] - 30),
                color,
            )
        else:
            image = put_text(
                image,
                f"{class_names[labels[n]-1]}({scores[n]:.2f})",
                (boxes[n][0], boxes[n][1] - 30),
                color,
            )

    return image


# def apply_bbox(image, boxes, class_names):
#     COLORS = np.random.uniform(0, 255, size=(len(class_names) + 1, 3))

#     for n in range(len(boxes)):
#         label = boxes[n][0].cpu().item()
#         if label == 0:
#             continue
#         else:
#             color = COLORS[random.randrange(0, len(COLORS))]

#         x1, y1, x2, y2 = tuple(map(lambda x: round(x.cpu().item()), boxes[n][2:]))
#         score = boxes[n][1].cpu().item()
#         cv2.rectangle(
#             image,
#             (x1, y1),
#             (x2, y2),
#             color=color,
#             thickness=2,
#         )

#         image = put_text(
#             image,
#             f"{class_names[label-1]}({score:.2f})",
#             (x1, y1 - 30),
#             color,
#         )

#     return image
