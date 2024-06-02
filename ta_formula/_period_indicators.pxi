cimport cython
cimport numpy as np
from cython cimport view
import numpy as pynp

@cython.freelist(2)
cdef class _CalcPeriodIndicator:
    cdef int[:] _start_point_indi(self):
        pass

    cdef double[:] _basic_indi(self):
        pass

    cdef double[:] _period_indi(self, int p_begin, int p_end, double[:] basic_indi):
        pass

    cdef int _get_data_length(self):
        pass

    cdef np.ndarray calc_period_indi(self, int period_nums):
        cdef:
            int[:] idx
            int ilen, data_length, i
            np.ndarray ret
            double[:] inds, ret_view
            int p_begin, p_end
        idx = self._start_point_indi()
        ilen = idx.shape[0]
        if period_nums > 0 and period_nums <= ilen:
            idx = idx[-period_nums:]

        data_length = self._get_data_length()
        ret = make_double_array(data_length, data_length)
        ret_view = ret
        p_begin = idx[0]
        if p_begin >= data_length:
            return ret
        inds = self._basic_indi()

        for i in range(ilen):
            p_begin = idx[i]
            if p_begin >= data_length:
                break
            if i == ilen - 1:
                ret_view[p_begin:] = self._period_indi(p_begin, -1, inds)
            else:
                p_end = idx[i + 1]
                ret_view[p_begin:p_end] = self._period_indi(p_begin, p_end, inds)
        return ret

cdef class PeriodMaxBias(_CalcPeriodIndicator):
    cdef np.ndarray close
    cdef int ma_timeperiod

    def __cinit__(self, np.ndarray close, int ma_timeperiod):
        self.close = close
        self.ma_timeperiod = ma_timeperiod

    cdef int[:] _start_point_indi(self):
        ''' 计算所有拐点, 确认区间 '''
        cdef:
            int i,j
            np.ndarray ma250, ma250_1, ma250_2, v
            double[:] v_view
            int[:] indexes
        ma250 = SMA(self.close, self.ma_timeperiod)
        ma250_1 = np_shift(ma250, 1)
        ma250_2 = np_shift(ma250, 2)
        v = (ma250 - ma250_1) * (ma250_1 - ma250_2)
        v_view = v
        indexes = view.array(shape=(v.shape[0],), itemsize=sizeof(int), format="i")
        j = 0
        for i in range(v.shape[0]):
            if v_view[i] <= 0:
                indexes[j] = i+1
                j+=1
        return indexes[:j]

    cdef double[:] _basic_indi(self):
        return BIAS(self.close, self.ma_timeperiod)

    cdef double[:] _period_indi(self, int p_begin, int p_end, double[:] basic_indi):
        ''' 计算区间最大BIAS '''
        cdef double[:] _bias
        if p_end == -1:
            _bias = basic_indi[p_begin:]
        else:
            _bias = basic_indi[p_begin:p_end]
        return pynp.maximum.accumulate(_bias)

    cdef int _get_data_length(self):
        return self.close.shape[0]