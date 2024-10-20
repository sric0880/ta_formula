
#cython: language_level=3str

# THIS IS AUTO GENERATED FILE, DO NOT MODIFY THIS FILE
# USE `autogen_pyd.py` FOR UPDATING

cimport numpy as np

ctypedef (double, double) tuple_double2
ctypedef (double, double, double) tuple_double3
ctypedef (double, double, double, double) tuple_double4

ctypedef fused numeric_dtype:
    int
    double
    long long



#############################################
# START FUNCTIONS COPY FROM '_bool_func.pxi'

cdef inline v(double[:] line, int offset):
    """V型拐点"""
    cdef double a, b, c
    a, b, c = line[offset - 2], line[offset - 1], line[offset]
    return b <= a and b < c


cdef inline vdown(double[:] line, int offset):
    """/\型拐点"""
    cdef double a, b, c
    a, b, c = line[offset - 2], line[offset - 1], line[offset]
    return b >= a and b > c


cdef inline kup(double[:] line, int offset):
    """斜率向上"""
    return line[offset] > line[offset - 1]


cdef inline kdown(double[:] line, int offset):
    """斜率向下"""
    return line[offset] < line[offset - 1]


cdef inline ne_l(double[:] line1, double[:] line2, int offset):
    """line1[offset] != line2[offset]"""
    return line1[offset] != line2[offset]


cdef inline eq_l(double[:] line1, double[:] line2, int offset):
    """line1[offset] == line2[offset]"""
    return line1[offset] == line2[offset]


cdef inline lt_l(double[:] line1, double[:] line2, int offset):
    """line1[offset] < line2[offset]"""
    return line1[offset] < line2[offset]


cdef inline gt_l(double[:] line1, double[:] line2):
    """line1[offset] > line2[offset]"""
    return line1[-1] > line2[-1]


cdef inline lte_l(double[:] line1, double[:] line2, int offset):
    """line1[offset] <= line2[offset]"""
    return line1[offset] <= line2[offset]


cdef inline gte_l(double[:] line1, double[:] line2, int offset):
    """line1[offset] >= line2[offset]"""
    return line1[offset] >= line2[offset]


cdef inline ne(double[:] line, double num, int offset):
    """line[offset] != num"""
    return line[offset] != num


cdef inline eq(double[:] line, double num, int offset):
    """line[offset] == num"""
    return line[offset] == num


cdef inline lt(double[:] line, double num, int offset):
    """line[offset] < num"""
    return line[offset] < num


cdef inline gt(double[:] line, double num, int offset):
    """line[offset] > num"""
    return line[offset] > num

cdef inline lte(double[:] line, double num, int offset):
    """line[offset] <= num"""
    return line[offset] <= num


cdef inline gte(double[:] line, double num, int offset):
    """line[offset] >= num"""
    return line[offset] >= num


cdef inline crossup(double[:] line1, double[:] line2, int offset):
    """金叉: line1 上穿 line2"""
    cdef int pre_offset = offset - 1
    return (line1[offset] > line2[offset]) and (line1[pre_offset] <= line2[pre_offset])


cdef inline crossdown(double[:] line1, double[:] line2, int offset):
    """死叉: line1 下穿 line2"""
    cdef int pre_offset = offset - 1
    return (line1[offset] < line2[offset]) and (line1[pre_offset] >= line2[pre_offset])


cdef inline crossup_value(double[:] line, double value, int offset):
    """上穿value"""
    return (line[offset] > value) and (line[offset - 1] <= value)


cdef inline crossdown_value(double[:] line, double value, int offset):
    """下穿value"""
    return (line[offset] < value) and (line[offset - 1] >= value)


cdef inline openup(double[:] line1, double[:] line2, int offset):
    """开口增大"""
    cdef:
        int pre_offset
        double n1, n2, m1, m2
    pre_offset = offset - 1
    n1, n2 = line1[offset], line1[pre_offset]
    m1, m2 = line2[offset], line2[pre_offset]
    if n1 > m1 and n2 > m2:
        return (n1 - m1) > (n2 - m2)
    elif n1 < m1 and n2 < m2:
        return (m1 - n1) > (m2 - n2)
    else:
        return False


cdef inline opendown(double[:] line1, double[:] line2, int offset):
    """开口缩小"""
    cdef:
        int pre_offset
        double n1, n2, m1, m2
    pre_offset = offset - 1
    n1, n2 = line1[offset], line1[pre_offset]
    m1, m2 = line2[offset], line2[pre_offset]
    if n1 > m1 and n2 > m2:
        return (n1 - m1) < (n2 - m2)
    elif n1 < m1 and n2 < m2:
        return (m1 - n1) < (m2 - n2)
    else:
        return False


cdef inline red_bar(double[:] OPEN, double[:] CLOSE, int offset):
    """ 阳线，如果是十字星并且今收>昨收，也当成阳线 """
    cdef double o, c, pre_c
    o, c, pre_c = OPEN[offset], CLOSE[offset], CLOSE[offset - 1]
    return c > o or (c == o and c > pre_c)


cdef inline red_bar_real(double[:] OPEN, double[:] CLOSE, int offset):
    """ 上涨线 """
    cdef double o, c, pre_c
    o, c, pre_c = OPEN[offset], CLOSE[offset], CLOSE[offset - 1]
    return c > pre_c or (c == pre_c and c > o)


cdef inline green_bar(double[:] OPEN, double[:] CLOSE, int offset):
    """ 阴线，如果是十字星并且今收<=昨收，也当成阴线 """
    cdef double o, c, pre_c
    o, c, pre_c = OPEN[offset], CLOSE[offset], CLOSE[offset - 1]
    return c < o or (c == o and c <= pre_c)


cdef inline green_bar_real(double[:] OPEN, double[:] CLOSE, int offset):
    """ 下跌线 """
    cdef double o, c, pre_c
    o, c, pre_c = OPEN[offset], CLOSE[offset], CLOSE[offset - 1]
    return c < pre_c or (c == pre_c and c <= o)

# END FUNCTIONS COPY FROM '_bool_func.pxi'
#############################################




#############################################
# START FUNCTIONS GEN FROM '_ta_lib_stream.pxi'

cdef stream_ACOS( np.ndarray real )
cdef stream_AD( np.ndarray high , np.ndarray low , np.ndarray close , np.ndarray volume )
cdef stream_ADD( np.ndarray real0 , np.ndarray real1 )
cdef stream_ADOSC( np.ndarray high , np.ndarray low , np.ndarray close , np.ndarray volume , int fastperiod, int slowperiod)
cdef stream_ADX( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef stream_ADXR( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef stream_APO( np.ndarray real , int fastperiod, int slowperiod, int matype)
cdef stream_AROON( np.ndarray high , np.ndarray low , int timeperiod)
cdef stream_AROONOSC( np.ndarray high , np.ndarray low , int timeperiod)
cdef stream_ASIN( np.ndarray real )
cdef stream_ATAN( np.ndarray real )
cdef stream_ATR( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef stream_AVGPRICE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_BBANDS( np.ndarray real , int timeperiod, double nbdevup, double nbdevdn, int matype)
cdef stream_BETA( np.ndarray real0 , np.ndarray real1 , int timeperiod)
cdef stream_BOP( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CCI( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef stream_CDL2CROWS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDL3BLACKCROWS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDL3INSIDE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDL3LINESTRIKE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDL3OUTSIDE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDL3STARSINSOUTH( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDL3WHITESOLDIERS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLABANDONEDBABY( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef stream_CDLADVANCEBLOCK( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLBELTHOLD( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLBREAKAWAY( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLCLOSINGMARUBOZU( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLCONCEALBABYSWALL( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLCOUNTERATTACK( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLDARKCLOUDCOVER( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef stream_CDLDOJI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLDOJISTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLDRAGONFLYDOJI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLENGULFING( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLEVENINGDOJISTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef stream_CDLEVENINGSTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef stream_CDLGAPSIDESIDEWHITE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLGRAVESTONEDOJI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLHAMMER( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLHANGINGMAN( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLHARAMI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLHARAMICROSS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLHIGHWAVE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLHIKKAKE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLHIKKAKEMOD( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLHOMINGPIGEON( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLIDENTICAL3CROWS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLINNECK( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLINVERTEDHAMMER( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLKICKING( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLKICKINGBYLENGTH( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLLADDERBOTTOM( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLLONGLEGGEDDOJI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLLONGLINE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLMARUBOZU( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLMATCHINGLOW( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLMATHOLD( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef stream_CDLMORNINGDOJISTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef stream_CDLMORNINGSTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef stream_CDLONNECK( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLPIERCING( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLRICKSHAWMAN( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLRISEFALL3METHODS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLSEPARATINGLINES( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLSHOOTINGSTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLSHORTLINE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLSPINNINGTOP( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLSTALLEDPATTERN( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLSTICKSANDWICH( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLTAKURI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLTASUKIGAP( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLTHRUSTING( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLTRISTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLUNIQUE3RIVER( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLUPSIDEGAP2CROWS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CDLXSIDEGAP3METHODS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_CEIL( np.ndarray real )
cdef stream_CMO( np.ndarray real , int timeperiod)
cdef stream_CORREL( np.ndarray real0 , np.ndarray real1 , int timeperiod)
cdef stream_COS( np.ndarray real )
cdef stream_COSH( np.ndarray real )
cdef stream_DEMA( np.ndarray real , int timeperiod)
cdef stream_DIV( np.ndarray real0 , np.ndarray real1 )
cdef stream_DX( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef stream_EMA( np.ndarray real , int timeperiod)
cdef stream_EXP( np.ndarray real )
cdef stream_FLOOR( np.ndarray real )
cdef stream_HT_DCPERIOD( np.ndarray real )
cdef stream_HT_DCPHASE( np.ndarray real )
cdef stream_HT_PHASOR( np.ndarray real )
cdef stream_HT_SINE( np.ndarray real )
cdef stream_HT_TRENDLINE( np.ndarray real )
cdef stream_HT_TRENDMODE( np.ndarray real )
cdef stream_KAMA( np.ndarray real , int timeperiod)
cdef stream_LINEARREG( np.ndarray real , int timeperiod)
cdef stream_LINEARREG_ANGLE( np.ndarray real , int timeperiod)
cdef stream_LINEARREG_INTERCEPT( np.ndarray real , int timeperiod)
cdef stream_LINEARREG_SLOPE( np.ndarray real , int timeperiod)
cdef stream_LN( np.ndarray real )
cdef stream_LOG10( np.ndarray real )
cdef stream_MA( np.ndarray real , int timeperiod, int matype)
cdef tuple_double3 stream_MACD( np.ndarray real , int fastperiod, int slowperiod, int signalperiod)
cdef stream_MACDEXT( np.ndarray real , int fastperiod, int fastmatype, int slowperiod, int slowmatype, int signalperiod, int signalmatype)
cdef stream_MACDFIX( np.ndarray real , int signalperiod)
cdef stream_MAMA( np.ndarray real , double fastlimit, double slowlimit)
cdef stream_MAVP( np.ndarray real , np.ndarray periods , int minperiod, int maxperiod, int matype)
cdef stream_MAX( np.ndarray real , int timeperiod)
cdef stream_MAXINDEX( np.ndarray real , int timeperiod)
cdef stream_MEDPRICE( np.ndarray high , np.ndarray low )
cdef stream_MFI( np.ndarray high , np.ndarray low , np.ndarray close , np.ndarray volume , int timeperiod)
cdef stream_MIDPOINT( np.ndarray real , int timeperiod)
cdef stream_MIDPRICE( np.ndarray high , np.ndarray low , int timeperiod)
cdef stream_MIN( np.ndarray real , int timeperiod)
cdef stream_MININDEX( np.ndarray real , int timeperiod)
cdef stream_MINMAX( np.ndarray real , int timeperiod)
cdef stream_MINMAXINDEX( np.ndarray real , int timeperiod)
cdef stream_MINUS_DI( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef stream_MINUS_DM( np.ndarray high , np.ndarray low , int timeperiod)
cdef stream_MOM( np.ndarray real , int timeperiod)
cdef stream_MULT( np.ndarray real0 , np.ndarray real1 )
cdef stream_NATR( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef stream_OBV( np.ndarray real , np.ndarray volume )
cdef stream_PLUS_DI( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef stream_PLUS_DM( np.ndarray high , np.ndarray low , int timeperiod)
cdef stream_PPO( np.ndarray real , int fastperiod, int slowperiod, int matype)
cdef stream_ROC( np.ndarray real , int timeperiod)
cdef stream_ROCP( np.ndarray real , int timeperiod)
cdef stream_ROCR( np.ndarray real , int timeperiod)
cdef stream_ROCR100( np.ndarray real , int timeperiod)
cdef stream_RSI( np.ndarray real , int timeperiod)
cdef stream_SAR( np.ndarray high , np.ndarray low , double acceleration, double maximum)
cdef stream_SAREXT( np.ndarray high , np.ndarray low , double startvalue, double offsetonreverse, double accelerationinitlong, double accelerationlong, double accelerationmaxlong, double accelerationinitshort, double accelerationshort, double accelerationmaxshort)
cdef stream_SIN( np.ndarray real )
cdef stream_SINH( np.ndarray real )
cdef stream_SMA( np.ndarray real , int timeperiod)
cdef stream_SQRT( np.ndarray real )
cdef stream_STDDEV( np.ndarray real , int timeperiod, double nbdev)
cdef tuple_double2 stream_STOCH( np.ndarray high , np.ndarray low , np.ndarray close , int fastk_period, int slowk_period, int slowk_matype, int slowd_period, int slowd_matype)
cdef stream_STOCHF( np.ndarray high , np.ndarray low , np.ndarray close , int fastk_period, int fastd_period, int fastd_matype)
cdef stream_STOCHRSI( np.ndarray real , int timeperiod, int fastk_period, int fastd_period, int fastd_matype)
cdef stream_SUB( np.ndarray real0 , np.ndarray real1 )
cdef stream_SUM( np.ndarray real , int timeperiod)
cdef stream_T3( np.ndarray real , int timeperiod, double vfactor)
cdef stream_TAN( np.ndarray real )
cdef stream_TANH( np.ndarray real )
cdef stream_TEMA( np.ndarray real , int timeperiod)
cdef stream_TRANGE( np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_TRIMA( np.ndarray real , int timeperiod)
cdef stream_TRIX( np.ndarray real , int timeperiod)
cdef stream_TSF( np.ndarray real , int timeperiod)
cdef stream_TYPPRICE( np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_ULTOSC( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod1, int timeperiod2, int timeperiod3)
cdef stream_VAR( np.ndarray real , int timeperiod, double nbdev)
cdef stream_WCLPRICE( np.ndarray high , np.ndarray low , np.ndarray close )
cdef stream_WILLR( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef stream_WMA( np.ndarray real , int timeperiod)

# END FUNCTIONS GEN FROM '_ta_lib_stream.pxi'
#############################################


#############################################
# START FUNCTIONS GEN FROM '_numpy_funcs.pxi'

cdef void shift_inplace(double[::1] arr, int num) noexcept nogil
cdef np.ndarray shift(np.ndarray arr, int num)
cdef void replace(double[::1] arr, double orig, double value) noexcept nogil
cdef void ffill(double[::1] arr) noexcept nogil
cdef np.ndarray rolling_sum(numeric_dtype[::1] arr, int window)

# END FUNCTIONS GEN FROM '_numpy_funcs.pxi'
#############################################


#############################################
# START FUNCTIONS GEN FROM '_ta_lib_func.pxi'

cdef np.ndarray make_double_array(np.npy_intp length, int lookback)
cdef np.ndarray make_int_array(np.npy_intp length, int lookback)
cdef ACOS( np.ndarray real )
cdef AD( np.ndarray high , np.ndarray low , np.ndarray close , np.ndarray volume )
cdef ADD( np.ndarray real0 , np.ndarray real1 )
cdef ADOSC( np.ndarray high , np.ndarray low , np.ndarray close , np.ndarray volume , int fastperiod, int slowperiod)
cdef ADX( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef ADXR( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef APO( np.ndarray real , int fastperiod, int slowperiod, int matype)
cdef AROON( np.ndarray high , np.ndarray low , int timeperiod)
cdef AROONOSC( np.ndarray high , np.ndarray low , int timeperiod)
cdef ASIN( np.ndarray real )
cdef ATAN( np.ndarray real )
cdef ATR( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef AVGPRICE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef BBANDS( np.ndarray real , int timeperiod, double nbdevup, double nbdevdn, int matype)
cdef BETA( np.ndarray real0 , np.ndarray real1 , int timeperiod)
cdef BOP( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CCI( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef CDL2CROWS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDL3BLACKCROWS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDL3INSIDE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDL3LINESTRIKE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDL3OUTSIDE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDL3STARSINSOUTH( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDL3WHITESOLDIERS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLABANDONEDBABY( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef CDLADVANCEBLOCK( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLBELTHOLD( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLBREAKAWAY( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLCLOSINGMARUBOZU( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLCONCEALBABYSWALL( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLCOUNTERATTACK( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLDARKCLOUDCOVER( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef CDLDOJI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLDOJISTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLDRAGONFLYDOJI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLENGULFING( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLEVENINGDOJISTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef CDLEVENINGSTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef CDLGAPSIDESIDEWHITE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLGRAVESTONEDOJI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLHAMMER( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLHANGINGMAN( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLHARAMI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLHARAMICROSS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLHIGHWAVE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLHIKKAKE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLHIKKAKEMOD( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLHOMINGPIGEON( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLIDENTICAL3CROWS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLINNECK( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLINVERTEDHAMMER( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLKICKING( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLKICKINGBYLENGTH( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLLADDERBOTTOM( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLLONGLEGGEDDOJI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLLONGLINE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLMARUBOZU( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLMATCHINGLOW( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLMATHOLD( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef CDLMORNINGDOJISTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef CDLMORNINGSTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration)
cdef CDLONNECK( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLPIERCING( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLRICKSHAWMAN( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLRISEFALL3METHODS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLSEPARATINGLINES( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLSHOOTINGSTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLSHORTLINE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLSPINNINGTOP( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLSTALLEDPATTERN( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLSTICKSANDWICH( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLTAKURI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLTASUKIGAP( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLTHRUSTING( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLTRISTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLUNIQUE3RIVER( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLUPSIDEGAP2CROWS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CDLXSIDEGAP3METHODS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close )
cdef CEIL( np.ndarray real )
cdef CMO( np.ndarray real , int timeperiod)
cdef CORREL( np.ndarray real0 , np.ndarray real1 , int timeperiod)
cdef COS( np.ndarray real )
cdef COSH( np.ndarray real )
cdef DEMA( np.ndarray real , int timeperiod)
cdef DIV( np.ndarray real0 , np.ndarray real1 )
cdef DX( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef EMA( np.ndarray real , int timeperiod)
cdef EXP( np.ndarray real )
cdef FLOOR( np.ndarray real )
cdef HT_DCPERIOD( np.ndarray real )
cdef HT_DCPHASE( np.ndarray real )
cdef HT_PHASOR( np.ndarray real )
cdef HT_SINE( np.ndarray real )
cdef HT_TRENDLINE( np.ndarray real )
cdef HT_TRENDMODE( np.ndarray real )
cdef KAMA( np.ndarray real , int timeperiod)
cdef LINEARREG( np.ndarray real , int timeperiod)
cdef LINEARREG_ANGLE( np.ndarray real , int timeperiod)
cdef LINEARREG_INTERCEPT( np.ndarray real , int timeperiod)
cdef LINEARREG_SLOPE( np.ndarray real , int timeperiod)
cdef LN( np.ndarray real )
cdef LOG10( np.ndarray real )
cdef MA( np.ndarray real , int timeperiod, int matype)
cdef MACD( np.ndarray real , int fastperiod, int slowperiod, int signalperiod)
cdef MACDEXT( np.ndarray real , int fastperiod, int fastmatype, int slowperiod, int slowmatype, int signalperiod, int signalmatype)
cdef MACDFIX( np.ndarray real , int signalperiod)
cdef MAMA( np.ndarray real , double fastlimit, double slowlimit)
cdef MAVP( np.ndarray real , np.ndarray periods , int minperiod, int maxperiod, int matype)
cdef MAX( np.ndarray real , int timeperiod)
cdef MAXINDEX( np.ndarray real , int timeperiod)
cdef MEDPRICE( np.ndarray high , np.ndarray low )
cdef MFI( np.ndarray high , np.ndarray low , np.ndarray close , np.ndarray volume , int timeperiod)
cdef MIDPOINT( np.ndarray real , int timeperiod)
cdef MIDPRICE( np.ndarray high , np.ndarray low , int timeperiod)
cdef MIN( np.ndarray real , int timeperiod)
cdef MININDEX( np.ndarray real , int timeperiod)
cdef MINMAX( np.ndarray real , int timeperiod)
cdef MINMAXINDEX( np.ndarray real , int timeperiod)
cdef MINUS_DI( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef MINUS_DM( np.ndarray high , np.ndarray low , int timeperiod)
cdef MOM( np.ndarray real , int timeperiod)
cdef MULT( np.ndarray real0 , np.ndarray real1 )
cdef NATR( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef OBV( np.ndarray real , np.ndarray volume )
cdef PLUS_DI( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef PLUS_DM( np.ndarray high , np.ndarray low , int timeperiod)
cdef PPO( np.ndarray real , int fastperiod, int slowperiod, int matype)
cdef ROC( np.ndarray real , int timeperiod)
cdef ROCP( np.ndarray real , int timeperiod)
cdef ROCR( np.ndarray real , int timeperiod)
cdef ROCR100( np.ndarray real , int timeperiod)
cdef RSI( np.ndarray real , int timeperiod)
cdef SAR( np.ndarray high , np.ndarray low , double acceleration, double maximum)
cdef SAREXT( np.ndarray high , np.ndarray low , double startvalue, double offsetonreverse, double accelerationinitlong, double accelerationlong, double accelerationmaxlong, double accelerationinitshort, double accelerationshort, double accelerationmaxshort)
cdef SIN( np.ndarray real )
cdef SINH( np.ndarray real )
cdef SMA( np.ndarray real , int timeperiod)
cdef SQRT( np.ndarray real )
cdef STDDEV( np.ndarray real , int timeperiod, double nbdev)
cdef STOCH( np.ndarray high , np.ndarray low , np.ndarray close , int fastk_period, int slowk_period, int slowk_matype, int slowd_period, int slowd_matype)
cdef STOCHF( np.ndarray high , np.ndarray low , np.ndarray close , int fastk_period, int fastd_period, int fastd_matype)
cdef STOCHRSI( np.ndarray real , int timeperiod, int fastk_period, int fastd_period, int fastd_matype)
cdef SUB( np.ndarray real0 , np.ndarray real1 )
cdef SUM( np.ndarray real , int timeperiod)
cdef T3( np.ndarray real , int timeperiod, double vfactor)
cdef TAN( np.ndarray real )
cdef TANH( np.ndarray real )
cdef TEMA( np.ndarray real , int timeperiod)
cdef TRANGE( np.ndarray high , np.ndarray low , np.ndarray close )
cdef TRIMA( np.ndarray real , int timeperiod)
cdef TRIX( np.ndarray real , int timeperiod)
cdef TSF( np.ndarray real , int timeperiod)
cdef TYPPRICE( np.ndarray high , np.ndarray low , np.ndarray close )
cdef ULTOSC( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod1, int timeperiod2, int timeperiod3)
cdef VAR( np.ndarray real , int timeperiod, double nbdev)
cdef WCLPRICE( np.ndarray high , np.ndarray low , np.ndarray close )
cdef WILLR( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod)
cdef WMA( np.ndarray real , int timeperiod)

# END FUNCTIONS GEN FROM '_ta_lib_func.pxi'
#############################################


#############################################
# START FUNCTIONS GEN FROM 'indicators.pyx'

cdef recent_SMA(double[::1] real, int timeperiod, int calc_length)
cdef BIAS(np.ndarray real, int timeperiod)
cdef double stream_BIAS(np.ndarray real, int timeperiod)
cdef recent_BIAS(double[::1] real, int timeperiod, int calc_length)
cdef recent_MACD(double[::1] real, int fastperiod, int slowperiod , int signalperiod, int calc_length )
cdef recent_STOCH(double[::1] high, double[::1] low, double[::1] close, int fastk_period, int slowk_period, int slowk_matype, int slowd_period, int slowd_matype, int calc_length)
cdef KD(np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period, int slowk_period, int slowd_period,)
cdef tuple_double2 stream_KD(np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period, int slowk_period, int slowd_period)
cdef recent_KD(double[::1] high, double[::1] low, double[::1] close, int fastk_period, int slowk_period, int slowd_period, int calc_length)
cdef KDJ(np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period, int slowk_period, int slowd_period)
cdef tuple_double3 stream_KDJ(np.ndarray high, np.ndarray low, np.ndarray close, int fastk_period, int slowk_period, int slowd_period)
cdef recent_KDJ(double[::1] high, double[::1] low, double[::1] close, int fastk_period, int slowk_period, int slowd_period, int calc_length)
cdef SLOW_KD(np.ndarray high, np.ndarray low, np.ndarray close , int fastk_period, int slowkd_period)
cdef tuple_double2 stream_SLOW_KD(double[::1] high, double[::1] low, double[::1] close , int fastk_period, int slowkd_period)
cdef recent_SLOW_KD(double[::1] high, double[::1] low, double[::1] close , int fastk_period, int slowkd_period, int calc_length)
cdef AMPLITUDE(np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod)
cdef double stream_AMPLITUDE(double[::1] high, double[::1] low, double[::1] close, int timeperiod)
cdef recent_AMPLITUDE(np.ndarray high, np.ndarray low, np.ndarray close, int timeperiod, int calc_length)
cdef ZIG(double[::1] real, double perctg)
cdef PERIOD_MAX_BIAS(np.ndarray close, int ma_timeperiod, int period_nums)

# END FUNCTIONS GEN FROM 'indicators.pyx'
#############################################
