def test_seed_everything_v1():
    from src.random_seed.v1 import seed_everything
    import numpy as np
    import torch

    seed_everything(0)
    random1 = np.random.random()
    tensor1 = torch.rand(1)

    seed_everything(0)
    random2 = np.random.random()
    tensor2 = torch.rand(1)

    assert random1 == random2
    assert tensor1 == tensor2


def test_seed_worker():
    pass


test_seed_worker()
# test_seed_everything_v1()
