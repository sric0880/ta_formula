# 这个版本和cython编译的版本性能差不多
# 直接使用_indicators模块下的np_shift
# 不依赖numba
import numba
import numpy as np


@numba.njit
def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result