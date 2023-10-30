import random
import numpy as np
import os
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = (
        True  # 이거 False해야 재현성이 유지되는 경우도 있음, True 시 자동으로 최적화 해줌
    )
    np.random.default_rng(seed)  # random 모듈의 Generator 클래스로 생성 시


def seed_worker(worker_id) -> None:
    # 추가로, 파이토치 데이터로더의 각 워커는 기본적으로 base_seed + worker_id 의 값으로 개별적으로 시드 세팅이 되기 때문에
    # 다음과 같이 worker_init_fn() 을 정의해 시드 세팅을 수동으로 해 주어야 한다.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
