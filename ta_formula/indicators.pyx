#cython: language_level=3str, embedsignature=True, binding=True
include "_ta_lib_common.pxi"
include "_unstable_periods.pxi"
include "_ta_lib_func.pxi"
include "_ta_lib_stream.pxi"
include "_period_indicators.pxi"
include "_numpy_funcs.pxi"


cdef int check_length2_new(double[::1] a1, double[::1] a2) except -1:
    cdef:
        int length
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    return length

cdef int check_length3_new(double[::1] a1, double[::1] a2, double[::1] a3) except -1:
    cdef:
        int length
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    if length != a3.shape[0]:
        raise Exception("input array lengths are different")
    return length

cdef int check_length4_new(double[::1] a1, double[::1] a2, double[::1] a3, double[::1] a4) except -1:
    cdef:
        int length
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    if length != a3.shape[0]:
        raise Exception("input array lengths are different")
    if length != a4.shape[0]:
        raise Exception("input array lengths are different")
    return length


@wraparound(False)
@boundscheck(False)
cdef recent_SMA(double[::1] real, int timeperiod, int calc_length):
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
        int length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray outreal
    if calc_length == 0:
        raise Exception(f'recent_SMA function failed with error: calc_length == 0')
    real_data = &real[0]
    length = real.shape[0]
    outreal = make_double_array(calc_length, 0)
    retCode = lib.TA_SMA( length - calc_length , length - 1 , real_data , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data) )
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


cdef double stream_BIAS(np.ndarray real, int timeperiod):
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


cdef recent_BIAS(double[::1] real, int timeperiod, int calc_length):
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
    length = real.shape[0]
    ma = recent_SMA(real, timeperiod, calc_length)
    return (real[length - calc_length: length] - ma) * 100 / ma


@wraparound(False)
@boundscheck(False)
cdef recent_MACD(double[::1] real, int fastperiod, int slowperiod , int signalperiod, int calc_length ):
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
        int length
        TA_RetCode retCode
        double* real_data
        int outbegidx
        int outnbelement
        np.ndarray dif
        np.ndarray dea
        np.ndarray outmacdhist
    if calc_length == 0:
        raise Exception(f'recent_MACD function failed with error: calc_length == 0')
    real_data = &real[0]
    length = real.shape[0]
    dif = make_double_array(calc_length, 0)
    dea = make_double_array(calc_length, 0)
    outmacdhist = make_double_array(calc_length, 0)
    _ta_set_unstable_period(FUNC_UNST_IDS.ID_EMA, calc_length+ema_unstable_periods[slowperiod])
    retCode = lib.TA_MACD(length - calc_length , length - 1 , real_data , fastperiod , slowperiod , signalperiod , &outbegidx , &outnbelement , <double *>(dif.data) , <double *>(dea.data) , <double *>(outmacdhist.data) )
    _ta_check_success("TA_MACD", retCode)
    outmacdhist *= 2  # MACD = (DIF-DEA) * 2
    return dif, dea, outmacdhist


@wraparound(False)
@boundscheck(False)
cdef recent_STOCH(double[::1] high, double[::1] low, double[::1] close, int fastk_period, int slowk_period, int slowk_matype, int slowd_period, int slowd_matype, int calc_length):
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
        int length
        TA_RetCode retCode
        double* high_data
        double* low_data
        double* close_data
        int outbegidx
        int outnbelement
        np.ndarray outslowk
        np.ndarray outslowd
    if calc_length == 0:
        raise Exception(f'recent_STOCH function failed with error: calc_length == 0')
    length = check_length3_new(high, low, close)
    high_data = &high[0]
    low_data = &low[0]
    close_data = &close[0]
    outslowk = make_double_array(calc_length, 0)
    outslowd = make_double_array(calc_length, 0)
    retCode = lib.TA_STOCH(length - calc_length , length - 1 , high_data , low_data , close_data , fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype , &outbegidx , &outnbelement , <double *>(outslowk.data), <double *>(outslowd.data))
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

cdef tuple_double2 stream_KD(np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period, int slowk_period, int slowd_period):
    """ stream_KD

    stream_KD (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 9
        slowk_period: 3
        slowd_period: 3
    Outputs:
        k: (double)
        d: (double)
    """
    cdef int kp = slowk_period * 2 - 1
    cdef int dp = slowd_period * 2 - 1
    _ta_set_unstable_period(FUNC_UNST_IDS.ID_EMA, ema_unstable_periods[kp] + ema_unstable_periods[dp])
    return stream_STOCH(high, low, close, fastk_period, kp, 1, dp, 1)

cdef recent_KD(double[::1] high, double[::1] low, double[::1] close, int fastk_period, int slowk_period, int slowd_period, int calc_length):
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
    return recent_STOCH(high, low, close, fastk_period, kp, 1, dp, 1, calc_length)


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

cdef tuple_double3 stream_KDJ(np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period, int slowk_period, int slowd_period):
    """ stream_KDJ

    stream_KDJ (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 9
        slowk_period: 3
        slowd_period: 3
    Outputs:
        k: (double)
        d: (double)
        j: (double)
    """
    cdef int kp = slowk_period * 2 - 1
    cdef int dp = slowd_period * 2 - 1
    _ta_set_unstable_period(FUNC_UNST_IDS.ID_EMA, ema_unstable_periods[kp] + ema_unstable_periods[dp])
    cdef double k, d
    k, d = stream_STOCH(high, low, close, fastk_period, kp, 1, dp, 1)
    return (k, d, (3 * k) - (2 * d))

cdef recent_KDJ(double[::1] high, double[::1] low, double[::1] close, int fastk_period, int slowk_period, int slowd_period, int calc_length):
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
    k, d = recent_STOCH(high, low, close, fastk_period, kp, 1, dp, 1, calc_length)
    return k, d, (3 * k) - (2 * d)


@wraparound(False)
@boundscheck(False)
cdef np.ndarray _stream_SLOW_K(double[::1] high, double[::1] low, double[::1] close , int fastk_period, int slowkd_period, int calc_length):
    cdef:
        int length
        np.npy_int begidx, endidx, newbegidx, newlength, lookback
        double* high_data
        double* low_data
        double* close_data
        TA_RetCode retCode 
        int outbegidx
        int outnbelement
        int unstable_period
        np.ndarray input1, input2
    unstable_period = ema_unstable_periods[slowkd_period]
    lookback = fastk_period + 2 * unstable_period + slowkd_period + slowkd_period + calc_length
    length = check_length3_new(high, low, close)
    high_data = &high[0]
    low_data = &low[0]
    close_data = &close[0]
    begidx = check_begidx3(length, high_data, low_data, close_data)
    endidx = <np.npy_int>length - begidx - 1
    newbegidx = endidx - lookback
    if newbegidx < 0:
        raise Exception(f'SLOW_KD function failed with error: length {endidx} is not enough for min lookback {lookback}.')
    newlength = lookback + 1

    # eup = ema_unstable_periods
    # fastk_period + eup[slowkd_period] + slowkd_period + eup[slowkd_period] + slowkd_period
    input1 = make_double_array(newlength, lib.TA_MIN_Lookback( fastk_period )) # input1 = lowv
    retCode = lib.TA_MIN( newbegidx , endidx , low_data+begidx , fastk_period , &outbegidx , &outnbelement , <double *>(input1.data))
    _ta_check_success("TA_MIN", retCode)

    input2 = make_double_array(newlength, lib.TA_MAX_Lookback( fastk_period )) # input2 = highv
    retCode = lib.TA_MAX( newbegidx , endidx , high_data+begidx , fastk_period , &outbegidx , &outnbelement , <double *>(input2.data))
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

cdef tuple_double2 stream_SLOW_KD(double[::1] high, double[::1] low, double[::1] close , int fastk_period, int slowkd_period):
    """ stream_SLOW_KD

    stream_SLOW_KD (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 9
        slowkd_period: 3
    Outputs:
        k: (double)
        d: (double)
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


cdef recent_SLOW_KD(double[::1] high, double[::1] low, double[::1] close , int fastk_period, int slowkd_period, int calc_length):
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
    return (high - low) / shift(close, timeperiod)

cdef double stream_AMPLITUDE(double[::1] high, double[::1] low, double[::1] close, int timeperiod):
    """stream_AMPLITUDE

    振幅：（最高价-最低价）/ 前N收盘价, 一般N为1

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 1
    Outputs:
        amplitude: (double)
    """
    return (high[-1] - low[-1]) / close[-1 - timeperiod]

cdef recent_AMPLITUDE(np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod, int calc_length):
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


@wraparound(False)
@boundscheck(False)
cdef ZIG(double[::1] real, double perctg):
    """ZIG

    又称wave波浪指标，当价格变化百分比超过`perctg`时转向

    Inputs:
        prices: ['real']
    Parameters:
        perctg: (double)价格变化百分比
    Outputs:
        points: (ndarray)拐点，非拐点用nan填充
    """
    cdef:
        int length
        int i
        np.ndarray points
        double min_close, max_close, min_point_price, max_point_price, c, last_price
        double[:] points_view
    length = real.shape[0]
    points = make_double_array(length, length)
    min_close = 0
    max_close = 0
    min_point_price = -1
    max_point_price = -1
    points_view = points
    for i in range(length):
        c = real[i]
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
    last_price = real[length-1]
    # 还没有确认转向，倒数第二个拐点不能确认
    # if max_close > -1 and max_close > last_price:
    #     points_view[?] = max_close
    # if min_close > -1 and min_close < last_price:
    #     points_view[?] = min_close
    points_view[length-1] = last_price
    return points


cdef PERIOD_MAX_BIAS(np.ndarray close, int ma_timeperiod, int period_nums):
    """PERIOD_MAX_BIAS

    区间最大BIAS，按一根均线拐头确认拐点，从拐点开始取区间最大BIAS值

    Inputs:
        prices: ['close']
    Parameters:
        ma_timeperiod: (int)均线周期大小
        period_nums: (int)返回的区间段数，1表示最近一段，0表示所有段
    Outputs:
        BIAS: (ndarray) 所有区间的BIAS值
    """
    return PeriodMaxBias(close, ma_timeperiod).calc_period_indi(period_nums)
