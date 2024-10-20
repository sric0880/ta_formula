cimport numpy as np
from cython import boundscheck, wraparound

from .indicators cimport numeric_dtype

cdef extern from "math.h":
    bint isnan(double x)

cdef void shift_inplace(double[::1] arr, int num) noexcept nogil:
    if num > 0:
        arr[num:] = arr[:-num]
        arr[:num] = NaN
    elif num < 0:
        arr[:num] = arr[-num:]
        arr[num:] = NaN


cdef np.ndarray shift(np.ndarray arr, int num):
    cdef np.ndarray outreal = PyArray_Copy(arr)
    cdef double[:] out_view = outreal
    if num > 0:
        out_view[num:] = out_view[:-num]
        out_view[:num] = NaN
    elif num < 0:
        out_view[:num] = out_view[-num:]
        out_view[num:] = NaN
    return outreal


@wraparound(False)
@boundscheck(False)
cdef void replace(double[::1] arr, double orig, double value) noexcept nogil:
    cdef int i
    for i in range(arr.shape[0]):
        if arr[i] == orig:
            arr[i] = value


@wraparound(False)
@boundscheck(False)
cdef void ffill(double[::1] arr) noexcept nogil:
    cdef int i
    cdef double val
    for i in range(1, arr.shape[0]):
        val = arr[i]
        if val != val: # val is np.nan
            arr[i] = arr[i-1]


@wraparound(False)
@boundscheck(False)
# https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html
cdef np.ndarray rolling_sum(numeric_dtype[::1] arr, int window):
    cdef int size, start, i, j
    cdef numeric_dtype head, tail
    size = arr.shape[0]
    cdef np.ndarray ret = make_double_array(size, 0)
    cdef double[:] result_view = ret
    if window > size:
        for i in range(size):
            result_view[i] = NaN
        return ret
    cdef double last_result = 0
    for i in range(0, size):
        start = i - window
        if start < 0:
            tail = 0
        else:
            tail = arr[start]
        head = arr[i]
        if numeric_dtype is double:
            if isnan(tail):
                tail = 0
            if isnan(head):
                head = 0
        last_result = result_view[i] = last_result + head - tail
    for i in range(size):
        if i < window-1:
            result_view[i] = NaN
        if numeric_dtype is double:
            if isnan(arr[i]):
                for j in range(i, i+window):
                    if j < size:
                        result_view[j] = NaN
    return ret
