from multiprocessing import Manager
from typing import Any
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import albumentations as A

class CachedDataset(Dataset):
    def __init__(self) -> None:
        """
        이미지를 메모리에 캐싱해둘 수 있는 데이터셋
        IO 시간을 줄일 수 있음
        (메모리 관리는 되지 않아, 캐싱시 남은 메모리가 없을 시 문제가 발생할 수 있으니 주의)
        """
        self.shared_cache = Manager().dict()

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def get_cache_data(self, path: str, caching_transform: A.Compose) -> np.ndarray:
        """
        캐싱된 데이터가 있으면 불러오고
        없으면 캐싱을 하면서 불러옴

        Args:
            path: 읽어올 파일 경로
            caching_transform: 해당 transform까지 진행하고 캐싱합니다

        Returns:
            img: caching_transform된 ndarray 이미지
        """
        img = self.shared_cache.get(path, None)

        # 만약 캐싱된 데이터가 없다면
        if img is None:
            # 일단 모든 이미지는 3ch로 읽어들임
            img = cv2.imread(path)

            # ImageNet pretrained weight가 RGB 채널 순서로 학습되어 미리 변환해줌
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = caching_transform(image=img)['image']

            self.shared_cache[path] = img

        return img
