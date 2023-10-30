def test_divide_list_evenly_v1():
    from src.image_eda.v1 import divide_list_evenly

    data_list = [1, 2, 3, 4, 5]

    result = divide_list_evenly(data_list, 1)
    assert result == [[1, 2, 3, 4, 5]]

    result = divide_list_evenly(data_list, 2)
    assert result == [[1, 2, 3], [4, 5]]

    result = divide_list_evenly(data_list, 3)
    assert result == [[1, 2], [3, 4], [5]]

    result = divide_list_evenly(data_list, 4)
    assert result == [[1, 2], [3], [4], [5]]

    result = divide_list_evenly(data_list, 5)
    assert result == [[1], [2], [3], [4], [5]]


def test_calc_img_mean_std_multiprocess():
    from glob import glob
    from src.image_eda.v1 import calc_img_mean_std_multiprocess

    X_list = sorted(
        glob(f"/data/ny_body_experiment/data/HOOD/02_PREPROCESSED_IMAGE_V1/*/*_X.png")
    )

    print(len(X_list))

    X_mean, X_std = calc_img_mean_std_multiprocess(X_list, 20)
    print(X_mean)
    print(X_std)


test_calc_img_mean_std_multiprocess()
test_divide_list_evenly_v1()
