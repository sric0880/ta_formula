'''
区间指标，按另一指标确认拐点，从拐点开始计算区间指标
'''
import numpy as np

from .indicators import *

__all__ = [
    'calc_period_indi',
    'PERIOD_MAX_BIAS',
]


def calc_period_indi(start_point_indi, basic_indi, period_indi, data_length, period_nums=1):
    idx = start_point_indi()
    if period_nums and period_nums <= len(idx):
        idx = idx[-period_nums:]
    ilen = len(idx)

    ret = np.empty((data_length,), dtype='float64')
    ret.fill(np.nan)
    p_begin = idx[0]
    if p_begin >= data_length:
        return ret
    inds = basic_indi()

    for i in range(ilen):
        p_begin = idx[i]
        if p_begin >= data_length:
            break
        if i == ilen - 1:
            ret[p_begin:] = period_indi(p_begin, None, inds)
        else:
            p_end = idx[i + 1]
            ret[p_begin:p_end] = period_indi(p_begin, p_end, inds)
    return ret


def PERIOD_MAX_BIAS(CLOSE, ma_timeperiod, period_nums=1):
    """
    区间最大BIAS，按SMA确认拐点，从拐点开始取区间最大BIAS值
    """

    def _start_point_indi():
        ''' 计算所有拐点, 确认区间 '''
        ma250 = SMA(CLOSE, ma_timeperiod)
        ma250_1 = np_shift(ma250, 1)
        ma250_2 = np_shift(ma250, 2)
        v = (ma250 - ma250_1) * (ma250_1 - ma250_2)
        return np.argwhere(v <= 0).flatten(order='C') + 1

    def _basic_indi():
        return BIAS(CLOSE, ma_timeperiod)

    def _period_indi(p_begin, p_end, bias):
        ''' 计算区间最大BIAS '''
        if p_end is None:
            _bias = bias[p_begin:]
        else:
            _bias = bias[p_begin:p_end]
        return np.maximum.accumulate(_bias)

    return calc_period_indi(
            _start_point_indi,
            _basic_indi,
            _period_indi,
            len(CLOSE),
            period_nums=period_nums
        )