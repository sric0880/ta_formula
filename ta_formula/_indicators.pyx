#cython: language_level=3str, embedsignature=True, binding=True
include "_ta_lib_common.pxi"
include "_unstable_periods.pxi"
include "_ta_lib_func.pxi"
include "_ta_lib_stream.pxi"

from cython import boundscheck, wraparound


cdef void np_shift_inplace(double[:] arr, int num):
    if num > 0:
        arr[num:] = arr[:-num]
        arr[:num] = NaN
    elif num < 0:
        arr[:num] = arr[-num:]
        arr[num:] = NaN

cdef np.ndarray np_shift(np.ndarray arr, int num):
    cdef np.ndarray outreal = PyArray_Copy(arr)
    cdef double[:] out_view = outreal
    if num > 0:
        out_view[num:] = out_view[:-num]
        out_view[:num] = NaN
    elif num < 0:
        out_view[:num] = out_view[-num:]
        out_view[num:] = NaN
    return outreal


@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef recent_SMA( np.ndarray real , int timeperiod, int calc_length):
    """ recent_SMA

    Simple Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
        calc_length: 1
    Outputs:
        real: (ndarray)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    if calc_length == 0:
        raise Exception(f'recent_SMA function failed with error: calc_length == 0')
    real_data = <double*>real.data
    length = real.shape[0]
    outreal = make_double_array(calc_length, 0)
    retCode = lib.TA_SMA( <int>(length) - calc_length , <int>(length) - 1 , real_data , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data) )
    _ta_check_success("TA_SMA", retCode)
    return outreal 


cdef BIAS(np.ndarray real, int timeperiod):
    """
    乖离率

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 20
    Outputs:
        bias: (ndarray)
    """
    ma = SMA(real, timeperiod)
    return (real - ma) * 100 / ma


cdef stream_BIAS(np.ndarray real, int timeperiod):
    """
    乖离率

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 20
    Outputs:
        bias: (double)
    """
    ma = stream_SMA(real, timeperiod)
    return (real[-1] - ma) * 100 / ma


cdef recent_BIAS(np.ndarray real, int timeperiod, int calc_length):
    """ recent_BIAS
    乖离率

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 20
        calc_length: 1
    Outputs:
        bias: (ndarray)
    """
    cdef:
        int length
        np.ndarray ma
        cdef double[:] real_view
    length = <int>real.shape[0]
    ma = recent_SMA(real, timeperiod, calc_length)
    real_view = real
    return (real_view[length - calc_length: length] - ma) * 100 / ma


@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef recent_MACD( np.ndarray real, int fastperiod, int slowperiod , int signalperiod, int calc_length ):
    """ recent_MACD

    Moving Average Convergence/Divergence (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        fastperiod: 12
        slowperiod: 26
        signalperiod: 9
        calc_length: 1
    Outputs:
        dif: (ndarray)
        dea: (ndarray)
        macdhist: (ndarray)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray dif
        np.ndarray dea
        np.ndarray outmacdhist
    real = check_array(real)
    if calc_length == 0:
        raise Exception(f'recent_MACD function failed with error: calc_length == 0')
    real_data = <double*>real.data
    length = real.shape[0]
    dif = make_double_array(calc_length, 0)
    dea = make_double_array(calc_length, 0)
    outmacdhist = make_double_array(calc_length, 0)
    _ta_set_unstable_period(FUNC_UNST_IDS.ID_EMA, calc_length+ema_unstable_periods[slowperiod])
    retCode = lib.TA_MACD( <int>(length) - calc_length , <int>(length) - 1 , real_data , fastperiod , slowperiod , signalperiod , &outbegidx , &outnbelement , <double *>(dif.data) , <double *>(dea.data) , <double *>(outmacdhist.data) )
    _ta_check_success("TA_MACD", retCode)
    outmacdhist *= 2  # MACD = (DIF-DEA) * 2
    return dif , dea , outmacdhist 


@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef recent_STOCH( np.ndarray high , np.ndarray low , np.ndarray close, int fastk_period, int slowk_period, int slowk_matype, int slowd_period, int slowd_matype, int calc_length):
    """ recent_STOCH(high, low, close[, fastk_period=?, slowk_period=?, slowk_matype=?, slowd_period=?, slowd_matype=?, calc_length=?])

    Stochastic (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 5
        slowk_period: 3
        slowk_matype: 0
        slowd_period: 3
        slowd_matype: 0
        calc_length: 1
    Outputs:
        slowk: (ndarray)
        slowd: (ndarray)
    """
    cdef:
        np.npy_intp length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outslowk
        np.ndarray outslowd
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    if calc_length == 0:
        raise Exception(f'recent_STOCH function failed with error: calc_length == 0')
    length = check_length3(high, low, close)
    high_data = <double*>high.data
    low_data = <double*>low.data
    close_data = <double*>close.data
    outslowk = make_double_array(calc_length, 0)
    outslowd = make_double_array(calc_length, 0)
    retCode = lib.TA_STOCH( <int>(length) - calc_length , <int>(length) - 1 , high_data , low_data , close_data , fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype , &outbegidx , &outnbelement , <double *>(outslowk.data), <double *>(outslowd.data))
    _ta_check_success("TA_STOCH", retCode)
    return outslowk , outslowd 


cdef KD(np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period, int slowk_period, int slowd_period,):
    """ KD

    KD (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 9
        slowk_period: 3
        slowd_period: 3
    Outputs:
        k: (ndarray)
        d: (ndarray)
    """
    return STOCH( high, low, close, fastk_period, slowk_period * 2 - 1, 1, slowd_period * 2 - 1, 1)

cdef stream_KD(np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period, int slowk_period, int slowd_period):
    """ stream_KD

    stream_KD (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 9
        slowk_period: 3
        slowd_period: 3
    Outputs:
        k: (float)
        d: (float)
    """
    cdef int kp = slowk_period * 2 - 1
    cdef int dp = slowd_period * 2 - 1
    _ta_set_unstable_period(FUNC_UNST_IDS.ID_EMA, ema_unstable_periods[kp] + ema_unstable_periods[dp])
    return stream_STOCH( high, low, close, fastk_period, kp, 1, dp, 1)

cdef recent_KD( np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period, int slowk_period, int slowd_period, int calc_length):
    """ recent_KD

    recent_KD (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 9
        slowk_period: 3
        slowd_period: 3
        calc_length: 1
    Outputs:
        k: (ndarray)
        d: (ndarray)
    """
    cdef int kp = slowk_period * 2 - 1
    cdef int dp = slowd_period * 2 - 1
    _ta_set_unstable_period(FUNC_UNST_IDS.ID_EMA, ema_unstable_periods[kp] + ema_unstable_periods[dp])
    return recent_STOCH( high, low, close, fastk_period, kp, 1, dp, 1, calc_length)


cdef KDJ(np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period, int slowk_period, int slowd_period):
    """ KDJ

    KDJ (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 9
        slowk_period: 3
        slowd_period: 3
    Outputs:
        k: (ndarray)
        d: (ndarray)
        j: (ndarray)
    """
    cdef np.ndarray k, d
    k, d = STOCH( high, low, close, fastk_period, slowk_period * 2 - 1, 1, slowd_period * 2 - 1, 1)
    return k, d, (3 * k) - (2 * d)

cdef stream_KDJ(np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period, int slowk_period, int slowd_period):
    """ stream_KDJ

    stream_KDJ (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 9
        slowk_period: 3
        slowd_period: 3
    Outputs:
        k: (float)
        d: (float)
        j: (float)
    """
    cdef int kp = slowk_period * 2 - 1
    cdef int dp = slowd_period * 2 - 1
    _ta_set_unstable_period(FUNC_UNST_IDS.ID_EMA, ema_unstable_periods[kp] + ema_unstable_periods[dp])
    cdef double k, d
    k, d = stream_STOCH( high, low, close, fastk_period, kp, 1, dp, 1)
    return k, d, (3 * k) - (2 * d)

cdef recent_KDJ(np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period, int slowk_period, int slowd_period, int calc_length):
    """ recent_KDJ

    recent_KDJ (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 9
        slowk_period: 3
        slowd_period: 3
        calc_length: 1
    Outputs:
        k: (ndarray)
        d: (ndarray)
        j: (ndarray)
    """
    cdef int kp = slowk_period * 2 - 1
    cdef int dp = slowd_period * 2 - 1
    _ta_set_unstable_period(FUNC_UNST_IDS.ID_EMA, ema_unstable_periods[kp] + ema_unstable_periods[dp])
    cdef np.ndarray k, d
    k, d = recent_STOCH( high, low, close, fastk_period, kp, 1, dp, 1, calc_length)
    return k, d, (3 * k) - (2 * d)


@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef np.ndarray _stream_SLOW_K(np.ndarray high, np.ndarray low, np.ndarray close , int fastk_period, int slowkd_period, int calc_length):
    cdef:
        np.npy_intp length
        np.npy_int begidx, endidx, newbegidx, newlength, lookback
        TA_RetCode retCode 
        int outbegidx
        int outnbelement
        int unstable_period
        np.ndarray input1, input2
    unstable_period = ema_unstable_periods[slowkd_period]
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    lookback = fastk_period + 2 * unstable_period + slowkd_period + slowkd_period + calc_length
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <np.npy_int>length - begidx - 1
    newbegidx = endidx - lookback
    if newbegidx < 0:
        raise Exception(f'SLOW_KD function failed with error: length {endidx} is not enough for min lookback {lookback}.')
    newlength = lookback + 1

    # eup = ema_unstable_periods
    # fastk_period + eup[slowkd_period] + slowkd_period + eup[slowkd_period] + slowkd_period
    input1 = make_double_array(newlength, lib.TA_MIN_Lookback( fastk_period )) # input1 = lowv
    retCode = lib.TA_MIN( newbegidx , endidx , <double *>(low.data)+begidx , fastk_period , &outbegidx , &outnbelement , <double *>(input1.data))
    _ta_check_success("TA_MIN", retCode)

    input2 = make_double_array(newlength, lib.TA_MAX_Lookback( fastk_period )) # input2 = highv
    retCode = lib.TA_MAX( newbegidx , endidx , <double *>(high.data)+begidx , fastk_period , &outbegidx , &outnbelement , <double *>(input2.data))
    _ta_check_success("TA_MAX", retCode)

    _ta_set_unstable_period(FUNC_UNST_IDS.ID_EMA, unstable_period)

    # eup[slowkd_period] + slowkd_period + eup[slowkd_period] + slowkd_period
    input1 = (close[<np.npy_int>length-newlength:] - input1) / (input2 - input1) * 100  # input1 = rsv
    begidx = check_begidx1(newlength, <double*>(input1.data))
    endidx = newlength - begidx - 1
    lookback = begidx + lib.TA_EMA_Lookback( slowkd_period )
    # input2 = ema_rsv 
    retCode = lib.TA_EMA( 0 , endidx , <double *>(input1.data)+begidx , slowkd_period , &outbegidx , &outnbelement , <double *>(input2.data)+lookback )
    _ta_check_success("TA_EMA", retCode)

    # eup[slowkd_period]+slowkd_period
    begidx = check_begidx1(newlength, <double*>(input2.data))
    endidx = newlength - begidx - 1
    lookback = begidx + lib.TA_EMA_Lookback( slowkd_period )
    # input1 = k
    retCode = lib.TA_EMA( 0 , endidx , <double *>(input2.data)+begidx , slowkd_period , &outbegidx , &outnbelement , <double *>(input1.data)+lookback )
    _ta_check_success("TA_EMA", retCode)

    return input1


cdef SLOW_KD(np.ndarray high, np.ndarray low, np.ndarray close , int fastk_period, int slowkd_period):
    """ SLOW_KD

    SLOW_KD (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 9
        slowkd_period: 3
    Outputs:
        k: (ndarray)
        d: (ndarray)
    """
    cdef np.ndarray lowv = MIN(low, fastk_period)
    cdef np.ndarray highv = MAX(high, fastk_period)
    cdef np.ndarray rsv = EMA((close - lowv) / (highv - lowv) * 100, slowkd_period)
    cdef np.ndarray k = EMA(rsv, slowkd_period)
    cdef np.ndarray d = SMA(k, slowkd_period)
    return k, d

cdef stream_SLOW_KD(np.ndarray high, np.ndarray low, np.ndarray close , int fastk_period, int slowkd_period):
    """ stream_SLOW_KD

    stream_SLOW_KD (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 9
        slowkd_period: 3
    Outputs:
        k: (float)
        d: (float)
    """
    cdef:
        int endidx
        np.ndarray k
        int outbegidx
        int outnbelement
        double d
    k = _stream_SLOW_K(high, low, close, fastk_period, slowkd_period, 1)
    endidx = <int>k.shape[0] - 1
    # slowkd_period
    d = NaN
    retCode = lib.TA_SMA(endidx , endidx , <double *>(k.data) , slowkd_period , &outbegidx , &outnbelement , &d)
    _ta_check_success("stream_SLOW_KD", retCode)
    return (k[endidx], d)


cdef recent_SLOW_KD(np.ndarray high, np.ndarray low, np.ndarray close , int fastk_period, int slowkd_period, int calc_length):
    """ recent_SLOW_KD

    recent_SLOW_KD (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 9
        slowkd_period: 3
        calc_length: 1
    Outputs:
        k: (ndarray)
        d: (ndarray)
    """
    cdef:
        int length
        np.ndarray k, d
        int outbegidx
        int outnbelement
    k = _stream_SLOW_K(high, low, close, fastk_period, slowkd_period, calc_length)
    length = <int>k.shape[0]
    # slowkd_period
    d = make_double_array(calc_length, 0)
    retCode = lib.TA_SMA(length - calc_length , length - 1 , <double *>(k.data) , slowkd_period , &outbegidx , &outnbelement , <double *>(d.data))
    _ta_check_success("recent_SLOW_KD", retCode)
    return k[length-calc_length:], d


cdef AMPLITUDE(np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod):
    """AMPLITUDE

    振幅：（最高价-最低价）/ 前N收盘价, 一般N为1

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 1
    Outputs:
        amplitude: (ndarray)
    """
    return (high - low) / np_shift(close, timeperiod)

cdef stream_AMPLITUDE(double[:] high, double[:] low, double[:] close, int timeperiod):
    """stream_AMPLITUDE

    振幅：（最高价-最低价）/ 前N收盘价, 一般N为1

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 1
    Outputs:
        amplitude: (float)
    """
    return (high[-1] - low[-1]) / close[-1 - timeperiod]

cdef recent_AMPLITUDE(np.ndarray[DTYPE_t, ndim=1] high, np.ndarray[DTYPE_t, ndim=1] low, np.ndarray[DTYPE_t, ndim=1] close, int timeperiod, int calc_length):
    """recent_AMPLITUDE

    振幅：（最高价-最低价）/ 前N收盘价, 一般N为1

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 1
        calc_length: 1
    Outputs:
        amplitude: (ndarray)
    """
    return (high[-calc_length:] - low[-calc_length:]) / close[-calc_length - timeperiod: -timeperiod]


@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ZIG(np.ndarray real, double perctg):
    """ZIG

    又称wave波浪指标，当价格变化百分比超过`perctg`时转向

    Inputs:
        prices: ['real']
    Parameters:
        perctg: (float)价格变化百分比
    Outputs:
        points: (ndarray)拐点，非拐点用nan填充
    """
    cdef:
        np.npy_intp length
        int i
        np.ndarray points
        double min_close, max_close, min_point_price, max_point_price, c, last_price
        double[:] real_view, points_view
    length = real.shape[0]
    points = make_double_array(length, length)
    min_close = 0
    max_close = 0
    min_point_price = -1
    max_point_price = -1
    real_view = real
    points_view = points
    for i in range(length):
        c = real_view[i]
        if min_point_price > 0 and c > min_point_price:
            points_view[i] = min_close
            max_close = c
            max_point_price = c * (1 - perctg)
            # print(f"确认小拐点{min_close} max_close{max_close} max_point_price{max_point_price}")
            min_close = min_point_price = -1
        elif max_point_price > 0 and c < max_point_price:
            points_view[i] = max_close
            min_close = c
            min_point_price = c * (1 + perctg)
            max_close = max_point_price = -1
            # print(f"确认大拐点{max_close} min_close{min_close} min_point_price{min_point_price}")
        if min_close > -1:
            if min_close == 0:
                min_close = c
                min_point_price = c * (1 + perctg)
            elif c < min_close:
                min_close = c
                min_point_price = c * (1 + perctg)
        if max_close > -1 and c > max_close:
            max_close = c
            max_point_price = c * (1 - perctg)
    last_price = real_view[length-1]
    # 还没有确认转向，倒数第二个拐点不能确认
    # if max_close > -1 and max_close > last_price:
    #     points_view[?] = max_close
    # if min_close > -1 and min_close < last_price:
    #     points_view[?] = min_close
    points_view[length-1] = last_price
    return points
