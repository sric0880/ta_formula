cimport numpy as np
from numpy import nan
from cython import boundscheck, wraparound

cdef double NaN = nan

cdef extern from "numpy/arrayobject.h":
    int PyArray_TYPE(np.ndarray)
    np.ndarray PyArray_EMPTY(int, np.npy_intp*, int, int)
    np.ndarray PyArray_Copy(np.ndarray)
    int PyArray_FLAGS(np.ndarray)
    np.ndarray PyArray_GETCONTIGUOUS(np.ndarray)

np.import_array() # Initialize the NumPy C API

cdef np.ndarray check_array(np.ndarray real):
    if PyArray_TYPE(real) != np.NPY_DOUBLE:
        raise Exception("input array type is not double")
    if real.ndim != 1:
        raise Exception("input array has wrong dimensions")
    if not (PyArray_FLAGS(real) & np.NPY_ARRAY_C_CONTIGUOUS):
        real = PyArray_GETCONTIGUOUS(real)
    return real

cdef np.npy_intp check_length2(np.ndarray a1, np.ndarray a2) except -1:
    cdef:
        np.npy_intp length
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    return length

cdef np.npy_intp check_length3(np.ndarray a1, np.ndarray a2, np.ndarray a3) except -1:
    cdef:
        np.npy_intp length
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    if length != a3.shape[0]:
        raise Exception("input array lengths are different")
    return length

cdef np.npy_intp check_length4(np.ndarray a1, np.ndarray a2, np.ndarray a3, np.ndarray a4) except -1:
    cdef:
        np.npy_intp length
    length = a1.shape[0]
    if length != a2.shape[0]:
        raise Exception("input array lengths are different")
    if length != a3.shape[0]:
        raise Exception("input array lengths are different")
    if length != a4.shape[0]:
        raise Exception("input array lengths are different")
    return length

cdef np.npy_int check_begidx1(np.npy_intp length, double* a1):
    cdef:
        double val
        int i
    for i in range(length):
        val = a1[i]
        if val != val:
            continue
        return i
    else:
        return length - 1

cdef np.npy_int check_begidx2(np.npy_intp length, double* a1, double* a2):
    cdef:
        double val
        int i
    for i in range(length):
        val = a1[i]
        if val != val:
            continue
        val = a2[i]
        if val != val:
            continue
        return i
    else:
        return length - 1

cdef np.npy_int check_begidx3(np.npy_intp length, double* a1, double* a2, double* a3):
    cdef:
        double val
        int i
    for i in range(length):
        val = a1[i]
        if val != val:
            continue
        val = a2[i]
        if val != val:
            continue
        val = a3[i]
        if val != val:
            continue
        return i
    else:
        return length - 1

cdef np.npy_int check_begidx4(np.npy_intp length, double* a1, double* a2, double* a3, double* a4):
    cdef:
        double val
        int i
    for i in range(length):
        val = a1[i]
        if val != val:
            continue
        val = a2[i]
        if val != val:
            continue
        val = a3[i]
        if val != val:
            continue
        val = a4[i]
        if val != val:
            continue
        return i
    else:
        return length - 1

cdef np.ndarray make_double_array(np.npy_intp length, int lookback):
    cdef:
        np.ndarray outreal
        double* outreal_data
        int i
    outreal = PyArray_EMPTY(1, &length, np.NPY_DOUBLE, np.NPY_ARRAY_DEFAULT)
    outreal_data = <double*>outreal.data
    for i in range(min(lookback, length)):
        outreal_data[i] = NaN
    return outreal

cdef np.ndarray make_int_array(np.npy_intp length, int lookback):
    cdef:
        np.ndarray outinteger
        int* outinteger_data
        int i
    outinteger = PyArray_EMPTY(1, &length, np.NPY_INT32, np.NPY_ARRAY_DEFAULT)
    outinteger_data = <int*>outinteger.data
    for i in range(min(lookback, length)):
        outinteger_data[i] = 0
    return outinteger


@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ACOS( np.ndarray real ):
    """ ACOS(real)

    Vector Trigonometric ACos (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ACOS_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ACOS( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ACOS", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef AD( np.ndarray high , np.ndarray low , np.ndarray close , np.ndarray volume ):
    """ AD(high, low, close, volume)

    Chaikin A/D Line (Volume Indicators)

    Inputs:
        prices: ['high', 'low', 'close', 'volume']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    volume = check_array(volume)
    length = check_length4(high, low, close, volume)
    begidx = check_begidx4(length, <double*>(high.data), <double*>(low.data), <double*>(close.data), <double*>(volume.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_AD_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_AD( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , <double *>(volume.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_AD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ADD( np.ndarray real0 , np.ndarray real1 ):
    """ ADD(real0, real1)

    Vector Arithmetic Add (Math Operators)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real0 = check_array(real0)
    real1 = check_array(real1)
    length = check_length2(real0, real1)
    begidx = check_begidx2(length, <double*>(real0.data), <double*>(real1.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ADD_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ADD( 0 , endidx , <double *>(real0.data)+begidx , <double *>(real1.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ADD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ADOSC( np.ndarray high , np.ndarray low , np.ndarray close , np.ndarray volume , int fastperiod, int slowperiod):
    """ ADOSC(high, low, close, volume[, fastperiod=?, slowperiod=?])

    Chaikin A/D Oscillator (Volume Indicators)

    Inputs:
        prices: ['high', 'low', 'close', 'volume']
    Parameters:
        fastperiod: 3
        slowperiod: 10
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    volume = check_array(volume)
    length = check_length4(high, low, close, volume)
    begidx = check_begidx4(length, <double*>(high.data), <double*>(low.data), <double*>(close.data), <double*>(volume.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ADOSC_Lookback( fastperiod , slowperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ADOSC( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , <double *>(volume.data)+begidx , fastperiod , slowperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ADOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ADX( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod):
    """ ADX(high, low, close[, timeperiod=?])

    Average Directional Movement Index (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ADX_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ADX( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ADX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ADXR( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod):
    """ ADXR(high, low, close[, timeperiod=?])

    Average Directional Movement Index Rating (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ADXR_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ADXR( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ADXR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef APO( np.ndarray real , int fastperiod, int slowperiod, int matype):
    """ APO(real[, fastperiod=?, slowperiod=?, matype=?])

    Absolute Price Oscillator (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        fastperiod: 12
        slowperiod: 26
        matype: 0 (Simple Moving Average)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_APO_Lookback( fastperiod , slowperiod , matype )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_APO( 0 , endidx , <double *>(real.data)+begidx , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_APO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef AROON( np.ndarray high , np.ndarray low , int timeperiod):
    """ AROON(high, low[, timeperiod=?])

    Aroon (Momentum Indicators)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        aroondown
        aroonup
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outaroondown
        np.ndarray outaroonup
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_AROON_Lookback( timeperiod )
    outaroondown = make_double_array(length, lookback)
    outaroonup = make_double_array(length, lookback)
    retCode = lib.TA_AROON( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outaroondown.data)+lookback , <double *>(outaroonup.data)+lookback )
    _ta_check_success("TA_AROON", retCode)
    return outaroondown , outaroonup 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef AROONOSC( np.ndarray high , np.ndarray low , int timeperiod):
    """ AROONOSC(high, low[, timeperiod=?])

    Aroon Oscillator (Momentum Indicators)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_AROONOSC_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_AROONOSC( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_AROONOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ASIN( np.ndarray real ):
    """ ASIN(real)

    Vector Trigonometric ASin (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ASIN_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ASIN( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ASIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ATAN( np.ndarray real ):
    """ ATAN(real)

    Vector Trigonometric ATan (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ATAN_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ATAN( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ATAN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ATR( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod):
    """ ATR(high, low, close[, timeperiod=?])

    Average True Range (Volatility Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ATR_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ATR( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ATR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef AVGPRICE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ AVGPRICE(open, high, low, close)

    Average Price (Price Transform)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_AVGPRICE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_AVGPRICE( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_AVGPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef BBANDS( np.ndarray real , int timeperiod, double nbdevup, double nbdevdn, int matype):
    """ BBANDS(real[, timeperiod=?, nbdevup=?, nbdevdn=?, matype=?])

    Bollinger Bands (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 5
        nbdevup: 2.0
        nbdevdn: 2.0
        matype: 0 (Simple Moving Average)
    Outputs:
        upperband
        middleband
        lowerband
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outrealupperband
        np.ndarray outrealmiddleband
        np.ndarray outreallowerband
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_BBANDS_Lookback( timeperiod , nbdevup , nbdevdn , matype )
    outrealupperband = make_double_array(length, lookback)
    outrealmiddleband = make_double_array(length, lookback)
    outreallowerband = make_double_array(length, lookback)
    retCode = lib.TA_BBANDS( 0 , endidx , <double *>(real.data)+begidx , timeperiod , nbdevup , nbdevdn , matype , &outbegidx , &outnbelement , <double *>(outrealupperband.data)+lookback , <double *>(outrealmiddleband.data)+lookback , <double *>(outreallowerband.data)+lookback )
    _ta_check_success("TA_BBANDS", retCode)
    return outrealupperband , outrealmiddleband , outreallowerband 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef BETA( np.ndarray real0 , np.ndarray real1 , int timeperiod):
    """ BETA(real0, real1[, timeperiod=?])

    Beta (Statistic Functions)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Parameters:
        timeperiod: 5
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real0 = check_array(real0)
    real1 = check_array(real1)
    length = check_length2(real0, real1)
    begidx = check_begidx2(length, <double*>(real0.data), <double*>(real1.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_BETA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_BETA( 0 , endidx , <double *>(real0.data)+begidx , <double *>(real1.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_BETA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef BOP( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ BOP(open, high, low, close)

    Balance Of Power (Momentum Indicators)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_BOP_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_BOP( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_BOP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CCI( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod):
    """ CCI(high, low, close[, timeperiod=?])

    Commodity Channel Index (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CCI_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_CCI( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_CCI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDL2CROWS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDL2CROWS(open, high, low, close)

    Two Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDL2CROWS_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDL2CROWS( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDL2CROWS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDL3BLACKCROWS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDL3BLACKCROWS(open, high, low, close)

    Three Black Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDL3BLACKCROWS_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDL3BLACKCROWS( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDL3BLACKCROWS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDL3INSIDE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDL3INSIDE(open, high, low, close)

    Three Inside Up/Down (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDL3INSIDE_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDL3INSIDE( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDL3INSIDE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDL3LINESTRIKE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDL3LINESTRIKE(open, high, low, close)

    Three-Line Strike  (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDL3LINESTRIKE_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDL3LINESTRIKE( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDL3LINESTRIKE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDL3OUTSIDE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDL3OUTSIDE(open, high, low, close)

    Three Outside Up/Down (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDL3OUTSIDE_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDL3OUTSIDE( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDL3OUTSIDE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDL3STARSINSOUTH( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDL3STARSINSOUTH(open, high, low, close)

    Three Stars In The South (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDL3STARSINSOUTH_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDL3STARSINSOUTH( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDL3STARSINSOUTH", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDL3WHITESOLDIERS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDL3WHITESOLDIERS(open, high, low, close)

    Three Advancing White Soldiers (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDL3WHITESOLDIERS_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDL3WHITESOLDIERS( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDL3WHITESOLDIERS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLABANDONEDBABY( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration):
    """ CDLABANDONEDBABY(open, high, low, close[, penetration=?])

    Abandoned Baby (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLABANDONEDBABY_Lookback( penetration )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLABANDONEDBABY( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , penetration , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLABANDONEDBABY", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLADVANCEBLOCK( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLADVANCEBLOCK(open, high, low, close)

    Advance Block (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLADVANCEBLOCK_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLADVANCEBLOCK( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLADVANCEBLOCK", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLBELTHOLD( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLBELTHOLD(open, high, low, close)

    Belt-hold (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLBELTHOLD_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLBELTHOLD( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLBELTHOLD", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLBREAKAWAY( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLBREAKAWAY(open, high, low, close)

    Breakaway (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLBREAKAWAY_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLBREAKAWAY( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLBREAKAWAY", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLCLOSINGMARUBOZU( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLCLOSINGMARUBOZU(open, high, low, close)

    Closing Marubozu (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLCLOSINGMARUBOZU_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLCLOSINGMARUBOZU( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLCLOSINGMARUBOZU", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLCONCEALBABYSWALL( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLCONCEALBABYSWALL(open, high, low, close)

    Concealing Baby Swallow (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLCONCEALBABYSWALL_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLCONCEALBABYSWALL( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLCONCEALBABYSWALL", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLCOUNTERATTACK( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLCOUNTERATTACK(open, high, low, close)

    Counterattack (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLCOUNTERATTACK_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLCOUNTERATTACK( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLCOUNTERATTACK", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLDARKCLOUDCOVER( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration):
    """ CDLDARKCLOUDCOVER(open, high, low, close[, penetration=?])

    Dark Cloud Cover (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.5
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLDARKCLOUDCOVER_Lookback( penetration )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLDARKCLOUDCOVER( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , penetration , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLDARKCLOUDCOVER", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLDOJI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLDOJI(open, high, low, close)

    Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLDOJI_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLDOJI( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLDOJI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLDOJISTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLDOJISTAR(open, high, low, close)

    Doji Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLDOJISTAR_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLDOJISTAR( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLDOJISTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLDRAGONFLYDOJI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLDRAGONFLYDOJI(open, high, low, close)

    Dragonfly Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLDRAGONFLYDOJI_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLDRAGONFLYDOJI( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLDRAGONFLYDOJI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLENGULFING( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLENGULFING(open, high, low, close)

    Engulfing Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLENGULFING_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLENGULFING( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLENGULFING", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLEVENINGDOJISTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration):
    """ CDLEVENINGDOJISTAR(open, high, low, close[, penetration=?])

    Evening Doji Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLEVENINGDOJISTAR_Lookback( penetration )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLEVENINGDOJISTAR( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , penetration , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLEVENINGDOJISTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLEVENINGSTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration):
    """ CDLEVENINGSTAR(open, high, low, close[, penetration=?])

    Evening Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLEVENINGSTAR_Lookback( penetration )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLEVENINGSTAR( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , penetration , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLEVENINGSTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLGAPSIDESIDEWHITE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLGAPSIDESIDEWHITE(open, high, low, close)

    Up/Down-gap side-by-side white lines (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLGAPSIDESIDEWHITE_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLGAPSIDESIDEWHITE( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLGAPSIDESIDEWHITE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLGRAVESTONEDOJI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLGRAVESTONEDOJI(open, high, low, close)

    Gravestone Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLGRAVESTONEDOJI_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLGRAVESTONEDOJI( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLGRAVESTONEDOJI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLHAMMER( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLHAMMER(open, high, low, close)

    Hammer (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLHAMMER_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLHAMMER( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLHAMMER", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLHANGINGMAN( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLHANGINGMAN(open, high, low, close)

    Hanging Man (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLHANGINGMAN_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLHANGINGMAN( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLHANGINGMAN", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLHARAMI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLHARAMI(open, high, low, close)

    Harami Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLHARAMI_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLHARAMI( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLHARAMI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLHARAMICROSS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLHARAMICROSS(open, high, low, close)

    Harami Cross Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLHARAMICROSS_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLHARAMICROSS( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLHARAMICROSS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLHIGHWAVE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLHIGHWAVE(open, high, low, close)

    High-Wave Candle (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLHIGHWAVE_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLHIGHWAVE( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLHIGHWAVE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLHIKKAKE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLHIKKAKE(open, high, low, close)

    Hikkake Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLHIKKAKE_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLHIKKAKE( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLHIKKAKE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLHIKKAKEMOD( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLHIKKAKEMOD(open, high, low, close)

    Modified Hikkake Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLHIKKAKEMOD_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLHIKKAKEMOD( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLHIKKAKEMOD", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLHOMINGPIGEON( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLHOMINGPIGEON(open, high, low, close)

    Homing Pigeon (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLHOMINGPIGEON_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLHOMINGPIGEON( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLHOMINGPIGEON", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLIDENTICAL3CROWS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLIDENTICAL3CROWS(open, high, low, close)

    Identical Three Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLIDENTICAL3CROWS_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLIDENTICAL3CROWS( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLIDENTICAL3CROWS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLINNECK( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLINNECK(open, high, low, close)

    In-Neck Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLINNECK_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLINNECK( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLINNECK", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLINVERTEDHAMMER( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLINVERTEDHAMMER(open, high, low, close)

    Inverted Hammer (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLINVERTEDHAMMER_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLINVERTEDHAMMER( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLINVERTEDHAMMER", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLKICKING( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLKICKING(open, high, low, close)

    Kicking (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLKICKING_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLKICKING( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLKICKING", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLKICKINGBYLENGTH( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLKICKINGBYLENGTH(open, high, low, close)

    Kicking - bull/bear determined by the longer marubozu (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLKICKINGBYLENGTH_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLKICKINGBYLENGTH( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLKICKINGBYLENGTH", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLLADDERBOTTOM( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLLADDERBOTTOM(open, high, low, close)

    Ladder Bottom (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLLADDERBOTTOM_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLLADDERBOTTOM( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLLADDERBOTTOM", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLLONGLEGGEDDOJI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLLONGLEGGEDDOJI(open, high, low, close)

    Long Legged Doji (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLLONGLEGGEDDOJI_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLLONGLEGGEDDOJI( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLLONGLEGGEDDOJI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLLONGLINE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLLONGLINE(open, high, low, close)

    Long Line Candle (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLLONGLINE_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLLONGLINE( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLLONGLINE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLMARUBOZU( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLMARUBOZU(open, high, low, close)

    Marubozu (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLMARUBOZU_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLMARUBOZU( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLMARUBOZU", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLMATCHINGLOW( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLMATCHINGLOW(open, high, low, close)

    Matching Low (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLMATCHINGLOW_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLMATCHINGLOW( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLMATCHINGLOW", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLMATHOLD( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration):
    """ CDLMATHOLD(open, high, low, close[, penetration=?])

    Mat Hold (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.5
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLMATHOLD_Lookback( penetration )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLMATHOLD( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , penetration , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLMATHOLD", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLMORNINGDOJISTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration):
    """ CDLMORNINGDOJISTAR(open, high, low, close[, penetration=?])

    Morning Doji Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLMORNINGDOJISTAR_Lookback( penetration )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLMORNINGDOJISTAR( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , penetration , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLMORNINGDOJISTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLMORNINGSTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close , double penetration):
    """ CDLMORNINGSTAR(open, high, low, close[, penetration=?])

    Morning Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Parameters:
        penetration: 0.3
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLMORNINGSTAR_Lookback( penetration )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLMORNINGSTAR( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , penetration , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLMORNINGSTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLONNECK( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLONNECK(open, high, low, close)

    On-Neck Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLONNECK_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLONNECK( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLONNECK", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLPIERCING( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLPIERCING(open, high, low, close)

    Piercing Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLPIERCING_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLPIERCING( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLPIERCING", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLRICKSHAWMAN( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLRICKSHAWMAN(open, high, low, close)

    Rickshaw Man (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLRICKSHAWMAN_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLRICKSHAWMAN( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLRICKSHAWMAN", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLRISEFALL3METHODS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLRISEFALL3METHODS(open, high, low, close)

    Rising/Falling Three Methods (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLRISEFALL3METHODS_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLRISEFALL3METHODS( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLRISEFALL3METHODS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLSEPARATINGLINES( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLSEPARATINGLINES(open, high, low, close)

    Separating Lines (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLSEPARATINGLINES_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLSEPARATINGLINES( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLSEPARATINGLINES", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLSHOOTINGSTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLSHOOTINGSTAR(open, high, low, close)

    Shooting Star (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLSHOOTINGSTAR_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLSHOOTINGSTAR( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLSHOOTINGSTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLSHORTLINE( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLSHORTLINE(open, high, low, close)

    Short Line Candle (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLSHORTLINE_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLSHORTLINE( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLSHORTLINE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLSPINNINGTOP( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLSPINNINGTOP(open, high, low, close)

    Spinning Top (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLSPINNINGTOP_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLSPINNINGTOP( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLSPINNINGTOP", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLSTALLEDPATTERN( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLSTALLEDPATTERN(open, high, low, close)

    Stalled Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLSTALLEDPATTERN_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLSTALLEDPATTERN( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLSTALLEDPATTERN", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLSTICKSANDWICH( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLSTICKSANDWICH(open, high, low, close)

    Stick Sandwich (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLSTICKSANDWICH_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLSTICKSANDWICH( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLSTICKSANDWICH", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLTAKURI( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLTAKURI(open, high, low, close)

    Takuri (Dragonfly Doji with very long lower shadow) (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLTAKURI_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLTAKURI( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLTAKURI", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLTASUKIGAP( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLTASUKIGAP(open, high, low, close)

    Tasuki Gap (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLTASUKIGAP_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLTASUKIGAP( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLTASUKIGAP", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLTHRUSTING( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLTHRUSTING(open, high, low, close)

    Thrusting Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLTHRUSTING_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLTHRUSTING( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLTHRUSTING", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLTRISTAR( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLTRISTAR(open, high, low, close)

    Tristar Pattern (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLTRISTAR_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLTRISTAR( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLTRISTAR", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLUNIQUE3RIVER( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLUNIQUE3RIVER(open, high, low, close)

    Unique 3 River (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLUNIQUE3RIVER_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLUNIQUE3RIVER( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLUNIQUE3RIVER", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLUPSIDEGAP2CROWS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLUPSIDEGAP2CROWS(open, high, low, close)

    Upside Gap Two Crows (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLUPSIDEGAP2CROWS_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLUPSIDEGAP2CROWS( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLUPSIDEGAP2CROWS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CDLXSIDEGAP3METHODS( np.ndarray open , np.ndarray high , np.ndarray low , np.ndarray close ):
    """ CDLXSIDEGAP3METHODS(open, high, low, close)

    Upside/Downside Gap Three Methods (Pattern Recognition)

    Inputs:
        prices: ['open', 'high', 'low', 'close']
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    open = check_array(open)
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length4(open, high, low, close)
    begidx = check_begidx4(length, <double*>(open.data), <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CDLXSIDEGAP3METHODS_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_CDLXSIDEGAP3METHODS( 0 , endidx , <double *>(open.data)+begidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_CDLXSIDEGAP3METHODS", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CEIL( np.ndarray real ):
    """ CEIL(real)

    Vector Ceil (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CEIL_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_CEIL( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_CEIL", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CMO( np.ndarray real , int timeperiod):
    """ CMO(real[, timeperiod=?])

    Chande Momentum Oscillator (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CMO_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_CMO( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_CMO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef CORREL( np.ndarray real0 , np.ndarray real1 , int timeperiod):
    """ CORREL(real0, real1[, timeperiod=?])

    Pearson's Correlation Coefficient (r) (Statistic Functions)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real0 = check_array(real0)
    real1 = check_array(real1)
    length = check_length2(real0, real1)
    begidx = check_begidx2(length, <double*>(real0.data), <double*>(real1.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_CORREL_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_CORREL( 0 , endidx , <double *>(real0.data)+begidx , <double *>(real1.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_CORREL", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef COS( np.ndarray real ):
    """ COS(real)

    Vector Trigonometric Cos (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_COS_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_COS( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_COS", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef COSH( np.ndarray real ):
    """ COSH(real)

    Vector Trigonometric Cosh (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_COSH_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_COSH( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_COSH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef DEMA( np.ndarray real , int timeperiod):
    """ DEMA(real[, timeperiod=?])

    Double Exponential Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_DEMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_DEMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_DEMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef DIV( np.ndarray real0 , np.ndarray real1 ):
    """ DIV(real0, real1)

    Vector Arithmetic Div (Math Operators)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real0 = check_array(real0)
    real1 = check_array(real1)
    length = check_length2(real0, real1)
    begidx = check_begidx2(length, <double*>(real0.data), <double*>(real1.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_DIV_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_DIV( 0 , endidx , <double *>(real0.data)+begidx , <double *>(real1.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_DIV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef DX( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod):
    """ DX(high, low, close[, timeperiod=?])

    Directional Movement Index (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_DX_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_DX( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_DX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef EMA( np.ndarray real , int timeperiod):
    """ EMA(real[, timeperiod=?])

    Exponential Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_EMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_EMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_EMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef EXP( np.ndarray real ):
    """ EXP(real)

    Vector Arithmetic Exp (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_EXP_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_EXP( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_EXP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef FLOOR( np.ndarray real ):
    """ FLOOR(real)

    Vector Floor (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_FLOOR_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_FLOOR( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_FLOOR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef HT_DCPERIOD( np.ndarray real ):
    """ HT_DCPERIOD(real)

    Hilbert Transform - Dominant Cycle Period (Cycle Indicators)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_HT_DCPERIOD_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_HT_DCPERIOD( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_HT_DCPERIOD", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef HT_DCPHASE( np.ndarray real ):
    """ HT_DCPHASE(real)

    Hilbert Transform - Dominant Cycle Phase (Cycle Indicators)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_HT_DCPHASE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_HT_DCPHASE( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_HT_DCPHASE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef HT_PHASOR( np.ndarray real ):
    """ HT_PHASOR(real)

    Hilbert Transform - Phasor Components (Cycle Indicators)

    Inputs:
        real: (any ndarray)
    Outputs:
        inphase
        quadrature
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinphase
        np.ndarray outquadrature
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_HT_PHASOR_Lookback( )
    outinphase = make_double_array(length, lookback)
    outquadrature = make_double_array(length, lookback)
    retCode = lib.TA_HT_PHASOR( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outinphase.data)+lookback , <double *>(outquadrature.data)+lookback )
    _ta_check_success("TA_HT_PHASOR", retCode)
    return outinphase , outquadrature 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef HT_SINE( np.ndarray real ):
    """ HT_SINE(real)

    Hilbert Transform - SineWave (Cycle Indicators)

    Inputs:
        real: (any ndarray)
    Outputs:
        sine
        leadsine
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outsine
        np.ndarray outleadsine
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_HT_SINE_Lookback( )
    outsine = make_double_array(length, lookback)
    outleadsine = make_double_array(length, lookback)
    retCode = lib.TA_HT_SINE( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outsine.data)+lookback , <double *>(outleadsine.data)+lookback )
    _ta_check_success("TA_HT_SINE", retCode)
    return outsine , outleadsine 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef HT_TRENDLINE( np.ndarray real ):
    """ HT_TRENDLINE(real)

    Hilbert Transform - Instantaneous Trendline (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_HT_TRENDLINE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_HT_TRENDLINE( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_HT_TRENDLINE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef HT_TRENDMODE( np.ndarray real ):
    """ HT_TRENDMODE(real)

    Hilbert Transform - Trend vs Cycle Mode (Cycle Indicators)

    Inputs:
        real: (any ndarray)
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_HT_TRENDMODE_Lookback( )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_HT_TRENDMODE( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_HT_TRENDMODE", retCode)
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef KAMA( np.ndarray real , int timeperiod):
    """ KAMA(real[, timeperiod=?])

    Kaufman Adaptive Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_KAMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_KAMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_KAMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef LINEARREG( np.ndarray real , int timeperiod):
    """ LINEARREG(real[, timeperiod=?])

    Linear Regression (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_LINEARREG_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_LINEARREG( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_LINEARREG", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef LINEARREG_ANGLE( np.ndarray real , int timeperiod):
    """ LINEARREG_ANGLE(real[, timeperiod=?])

    Linear Regression Angle (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_LINEARREG_ANGLE_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_LINEARREG_ANGLE( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_LINEARREG_ANGLE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef LINEARREG_INTERCEPT( np.ndarray real , int timeperiod):
    """ LINEARREG_INTERCEPT(real[, timeperiod=?])

    Linear Regression Intercept (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_LINEARREG_INTERCEPT_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_LINEARREG_INTERCEPT( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_LINEARREG_INTERCEPT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef LINEARREG_SLOPE( np.ndarray real , int timeperiod):
    """ LINEARREG_SLOPE(real[, timeperiod=?])

    Linear Regression Slope (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_LINEARREG_SLOPE_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_LINEARREG_SLOPE( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_LINEARREG_SLOPE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef LN( np.ndarray real ):
    """ LN(real)

    Vector Log Natural (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_LN_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_LN( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_LN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef LOG10( np.ndarray real ):
    """ LOG10(real)

    Vector Log10 (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_LOG10_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_LOG10( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_LOG10", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MA( np.ndarray real , int timeperiod, int matype):
    """ MA(real[, timeperiod=?, matype=?])

    Moving average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
        matype: 0 (Simple Moving Average)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MA_Lookback( timeperiod , matype )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , matype , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MACD( np.ndarray real , int fastperiod, int slowperiod, int signalperiod):
    """ MACD(real[, fastperiod=?, slowperiod=?, signalperiod=?])

    Moving Average Convergence/Divergence (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        fastperiod: 12
        slowperiod: 26
        signalperiod: 9
    Outputs:
        macd
        macdsignal
        macdhist
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outmacd
        np.ndarray outmacdsignal
        np.ndarray outmacdhist
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MACD_Lookback( fastperiod , slowperiod , signalperiod )
    outmacd = make_double_array(length, lookback)
    outmacdsignal = make_double_array(length, lookback)
    outmacdhist = make_double_array(length, lookback)
    retCode = lib.TA_MACD( 0 , endidx , <double *>(real.data)+begidx , fastperiod , slowperiod , signalperiod , &outbegidx , &outnbelement , <double *>(outmacd.data)+lookback , <double *>(outmacdsignal.data)+lookback , <double *>(outmacdhist.data)+lookback )
    _ta_check_success("TA_MACD", retCode)
    outmacdhist *= 2  # ADDED BY ME
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MACDEXT( np.ndarray real , int fastperiod, int fastmatype, int slowperiod, int slowmatype, int signalperiod, int signalmatype):
    """ MACDEXT(real[, fastperiod=?, fastmatype=?, slowperiod=?, slowmatype=?, signalperiod=?, signalmatype=?])

    MACD with controllable MA type (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        fastperiod: 12
        fastmatype: 0
        slowperiod: 26
        slowmatype: 0
        signalperiod: 9
        signalmatype: 0
    Outputs:
        macd
        macdsignal
        macdhist
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outmacd
        np.ndarray outmacdsignal
        np.ndarray outmacdhist
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MACDEXT_Lookback( fastperiod , fastmatype , slowperiod , slowmatype , signalperiod , signalmatype )
    outmacd = make_double_array(length, lookback)
    outmacdsignal = make_double_array(length, lookback)
    outmacdhist = make_double_array(length, lookback)
    retCode = lib.TA_MACDEXT( 0 , endidx , <double *>(real.data)+begidx , fastperiod , fastmatype , slowperiod , slowmatype , signalperiod , signalmatype , &outbegidx , &outnbelement , <double *>(outmacd.data)+lookback , <double *>(outmacdsignal.data)+lookback , <double *>(outmacdhist.data)+lookback )
    _ta_check_success("TA_MACDEXT", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MACDFIX( np.ndarray real , int signalperiod):
    """ MACDFIX(real[, signalperiod=?])

    Moving Average Convergence/Divergence Fix 12/26 (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        signalperiod: 9
    Outputs:
        macd
        macdsignal
        macdhist
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outmacd
        np.ndarray outmacdsignal
        np.ndarray outmacdhist
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MACDFIX_Lookback( signalperiod )
    outmacd = make_double_array(length, lookback)
    outmacdsignal = make_double_array(length, lookback)
    outmacdhist = make_double_array(length, lookback)
    retCode = lib.TA_MACDFIX( 0 , endidx , <double *>(real.data)+begidx , signalperiod , &outbegidx , &outnbelement , <double *>(outmacd.data)+lookback , <double *>(outmacdsignal.data)+lookback , <double *>(outmacdhist.data)+lookback )
    _ta_check_success("TA_MACDFIX", retCode)
    return outmacd , outmacdsignal , outmacdhist 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MAMA( np.ndarray real , double fastlimit, double slowlimit):
    """ MAMA(real[, fastlimit=?, slowlimit=?])

    MESA Adaptive Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        fastlimit: 0.5
        slowlimit: 0.05
    Outputs:
        mama
        fama
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outmama
        np.ndarray outfama
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MAMA_Lookback( fastlimit , slowlimit )
    outmama = make_double_array(length, lookback)
    outfama = make_double_array(length, lookback)
    retCode = lib.TA_MAMA( 0 , endidx , <double *>(real.data)+begidx , fastlimit , slowlimit , &outbegidx , &outnbelement , <double *>(outmama.data)+lookback , <double *>(outfama.data)+lookback )
    _ta_check_success("TA_MAMA", retCode)
    return outmama , outfama 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MAVP( np.ndarray real , np.ndarray periods , int minperiod, int maxperiod, int matype):
    """ MAVP(real, periods[, minperiod=?, maxperiod=?, matype=?])

    Moving average with variable period (Overlap Studies)

    Inputs:
        real: (any ndarray)
        periods: (any ndarray)
    Parameters:
        minperiod: 2
        maxperiod: 30
        matype: 0 (Simple Moving Average)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    periods = check_array(periods)
    length = check_length2(real, periods)
    begidx = check_begidx2(length, <double*>(real.data), <double*>(periods.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MAVP_Lookback( minperiod , maxperiod , matype )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MAVP( 0 , endidx , <double *>(real.data)+begidx , <double *>(periods.data)+begidx , minperiod , maxperiod , matype , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MAVP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MAX( np.ndarray real , int timeperiod):
    """ MAX(real[, timeperiod=?])

    Highest value over a specified period (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MAX_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MAX( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MAX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MAXINDEX( np.ndarray real , int timeperiod):
    """ MAXINDEX(real[, timeperiod=?])

    Index of highest value over a specified period (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MAXINDEX_Lookback( timeperiod )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_MAXINDEX( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_MAXINDEX", retCode)
    outinteger_data = <int*>outinteger.data
    for i from lookback <= i < length:
        outinteger_data[i] += begidx
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MEDPRICE( np.ndarray high , np.ndarray low ):
    """ MEDPRICE(high, low)

    Median Price (Price Transform)

    Inputs:
        prices: ['high', 'low']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MEDPRICE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MEDPRICE( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MEDPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MFI( np.ndarray high , np.ndarray low , np.ndarray close , np.ndarray volume , int timeperiod):
    """ MFI(high, low, close, volume[, timeperiod=?])

    Money Flow Index (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close', 'volume']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    volume = check_array(volume)
    length = check_length4(high, low, close, volume)
    begidx = check_begidx4(length, <double*>(high.data), <double*>(low.data), <double*>(close.data), <double*>(volume.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MFI_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MFI( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , <double *>(volume.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MFI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MIDPOINT( np.ndarray real , int timeperiod):
    """ MIDPOINT(real[, timeperiod=?])

    MidPoint over period (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MIDPOINT_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MIDPOINT( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MIDPOINT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MIDPRICE( np.ndarray high , np.ndarray low , int timeperiod):
    """ MIDPRICE(high, low[, timeperiod=?])

    Midpoint Price over period (Overlap Studies)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MIDPRICE_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MIDPRICE( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MIDPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MIN( np.ndarray real , int timeperiod):
    """ MIN(real[, timeperiod=?])

    Lowest value over a specified period (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MIN_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MIN( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MININDEX( np.ndarray real , int timeperiod):
    """ MININDEX(real[, timeperiod=?])

    Index of lowest value over a specified period (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        integer (values are -100, 0 or 100)
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outinteger
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MININDEX_Lookback( timeperiod )
    outinteger = make_int_array(length, lookback)
    retCode = lib.TA_MININDEX( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <int *>(outinteger.data)+lookback )
    _ta_check_success("TA_MININDEX", retCode)
    outinteger_data = <int*>outinteger.data
    for i from lookback <= i < length:
        outinteger_data[i] += begidx
    return outinteger 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MINMAX( np.ndarray real , int timeperiod):
    """ MINMAX(real[, timeperiod=?])

    Lowest and highest values over a specified period (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        min
        max
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outmin
        np.ndarray outmax
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MINMAX_Lookback( timeperiod )
    outmin = make_double_array(length, lookback)
    outmax = make_double_array(length, lookback)
    retCode = lib.TA_MINMAX( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outmin.data)+lookback , <double *>(outmax.data)+lookback )
    _ta_check_success("TA_MINMAX", retCode)
    return outmin , outmax 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MINMAXINDEX( np.ndarray real , int timeperiod):
    """ MINMAXINDEX(real[, timeperiod=?])

    Indexes of lowest and highest values over a specified period (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        minidx
        maxidx
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outminidx
        np.ndarray outmaxidx
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MINMAXINDEX_Lookback( timeperiod )
    outminidx = make_int_array(length, lookback)
    outmaxidx = make_int_array(length, lookback)
    retCode = lib.TA_MINMAXINDEX( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <int *>(outminidx.data)+lookback , <int *>(outmaxidx.data)+lookback )
    _ta_check_success("TA_MINMAXINDEX", retCode)
    outminidx_data = <int*>outminidx.data
    for i from lookback <= i < length:
        outminidx_data[i] += begidx
    outmaxidx_data = <int*>outmaxidx.data
    for i from lookback <= i < length:
        outmaxidx_data[i] += begidx
    return outminidx , outmaxidx 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MINUS_DI( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod):
    """ MINUS_DI(high, low, close[, timeperiod=?])

    Minus Directional Indicator (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MINUS_DI_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MINUS_DI( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MINUS_DI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MINUS_DM( np.ndarray high , np.ndarray low , int timeperiod):
    """ MINUS_DM(high, low[, timeperiod=?])

    Minus Directional Movement (Momentum Indicators)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MINUS_DM_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MINUS_DM( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MINUS_DM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MOM( np.ndarray real , int timeperiod):
    """ MOM(real[, timeperiod=?])

    Momentum (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MOM_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MOM( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MOM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef MULT( np.ndarray real0 , np.ndarray real1 ):
    """ MULT(real0, real1)

    Vector Arithmetic Mult (Math Operators)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real0 = check_array(real0)
    real1 = check_array(real1)
    length = check_length2(real0, real1)
    begidx = check_begidx2(length, <double*>(real0.data), <double*>(real1.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_MULT_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_MULT( 0 , endidx , <double *>(real0.data)+begidx , <double *>(real1.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_MULT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef NATR( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod):
    """ NATR(high, low, close[, timeperiod=?])

    Normalized Average True Range (Volatility Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_NATR_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_NATR( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_NATR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef OBV( np.ndarray real , np.ndarray volume ):
    """ OBV(real, volume)

    On Balance Volume (Volume Indicators)

    Inputs:
        real: (any ndarray)
        prices: ['volume']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    volume = check_array(volume)
    length = check_length2(real, volume)
    begidx = check_begidx2(length, <double*>(real.data), <double*>(volume.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_OBV_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_OBV( 0 , endidx , <double *>(real.data)+begidx , <double *>(volume.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_OBV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef PLUS_DI( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod):
    """ PLUS_DI(high, low, close[, timeperiod=?])

    Plus Directional Indicator (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_PLUS_DI_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_PLUS_DI( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_PLUS_DI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef PLUS_DM( np.ndarray high , np.ndarray low , int timeperiod):
    """ PLUS_DM(high, low[, timeperiod=?])

    Plus Directional Movement (Momentum Indicators)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_PLUS_DM_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_PLUS_DM( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_PLUS_DM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef PPO( np.ndarray real , int fastperiod, int slowperiod, int matype):
    """ PPO(real[, fastperiod=?, slowperiod=?, matype=?])

    Percentage Price Oscillator (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        fastperiod: 12
        slowperiod: 26
        matype: 0 (Simple Moving Average)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_PPO_Lookback( fastperiod , slowperiod , matype )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_PPO( 0 , endidx , <double *>(real.data)+begidx , fastperiod , slowperiod , matype , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_PPO", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ROC( np.ndarray real , int timeperiod):
    """ ROC(real[, timeperiod=?])

    Rate of change : ((real/prevPrice)-1)*100 (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ROC_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ROC( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ROC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ROCP( np.ndarray real , int timeperiod):
    """ ROCP(real[, timeperiod=?])

    Rate of change Percentage: (real-prevPrice)/prevPrice (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ROCP_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ROCP( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ROCP", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ROCR( np.ndarray real , int timeperiod):
    """ ROCR(real[, timeperiod=?])

    Rate of change ratio: (real/prevPrice) (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ROCR_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ROCR( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ROCR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ROCR100( np.ndarray real , int timeperiod):
    """ ROCR100(real[, timeperiod=?])

    Rate of change ratio 100 scale: (real/prevPrice)*100 (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 10
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ROCR100_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ROCR100( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ROCR100", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef RSI( np.ndarray real , int timeperiod):
    """ RSI(real[, timeperiod=?])

    Relative Strength Index (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_RSI_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_RSI( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_RSI", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef SAR( np.ndarray high , np.ndarray low , double acceleration, double maximum):
    """ SAR(high, low[, acceleration=?, maximum=?])

    Parabolic SAR (Overlap Studies)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        acceleration: 0.02
        maximum: 0.2
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SAR_Lookback( acceleration , maximum )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SAR( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , acceleration , maximum , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SAR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef SAREXT( np.ndarray high , np.ndarray low , double startvalue, double offsetonreverse, double accelerationinitlong, double accelerationlong, double accelerationmaxlong, double accelerationinitshort, double accelerationshort, double accelerationmaxshort):
    """ SAREXT(high, low[, startvalue=?, offsetonreverse=?, accelerationinitlong=?, accelerationlong=?, accelerationmaxlong=?, accelerationinitshort=?, accelerationshort=?, accelerationmaxshort=?])

    Parabolic SAR - Extended (Overlap Studies)

    Inputs:
        prices: ['high', 'low']
    Parameters:
        startvalue: 0.0
        offsetonreverse: 0.0
        accelerationinitlong: 0.02
        accelerationlong: 0.02
        accelerationmaxlong: 0.2
        accelerationinitshort: 0.02
        accelerationshort: 0.02
        accelerationmaxshort: 0.2
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    length = check_length2(high, low)
    begidx = check_begidx2(length, <double*>(high.data), <double*>(low.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SAREXT_Lookback( startvalue , offsetonreverse , accelerationinitlong , accelerationlong , accelerationmaxlong , accelerationinitshort , accelerationshort , accelerationmaxshort )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SAREXT( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , startvalue , offsetonreverse , accelerationinitlong , accelerationlong , accelerationmaxlong , accelerationinitshort , accelerationshort , accelerationmaxshort , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SAREXT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef SIN( np.ndarray real ):
    """ SIN(real)

    Vector Trigonometric Sin (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SIN_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SIN( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SIN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef SINH( np.ndarray real ):
    """ SINH(real)

    Vector Trigonometric Sinh (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SINH_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SINH( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SINH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef SMA( np.ndarray real , int timeperiod):
    """ SMA(real[, timeperiod=?])

    Simple Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef SQRT( np.ndarray real ):
    """ SQRT(real)

    Vector Square Root (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SQRT_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SQRT( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SQRT", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef STDDEV( np.ndarray real , int timeperiod, double nbdev):
    """ STDDEV(real[, timeperiod=?, nbdev=?])

    Standard Deviation (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 5
        nbdev: 1.0
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_STDDEV_Lookback( timeperiod , nbdev )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_STDDEV( 0 , endidx , <double *>(real.data)+begidx , timeperiod , nbdev , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_STDDEV", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef STOCH( np.ndarray high , np.ndarray low , np.ndarray close , int fastk_period, int slowk_period, int slowk_matype, int slowd_period, int slowd_matype):
    """ STOCH(high, low, close[, fastk_period=?, slowk_period=?, slowk_matype=?, slowd_period=?, slowd_matype=?])

    Stochastic (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 5
        slowk_period: 3
        slowk_matype: 0
        slowd_period: 3
        slowd_matype: 0
    Outputs:
        slowk
        slowd
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outslowk
        np.ndarray outslowd
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_STOCH_Lookback( fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype )
    outslowk = make_double_array(length, lookback)
    outslowd = make_double_array(length, lookback)
    retCode = lib.TA_STOCH( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , fastk_period , slowk_period , slowk_matype , slowd_period , slowd_matype , &outbegidx , &outnbelement , <double *>(outslowk.data)+lookback , <double *>(outslowd.data)+lookback )
    _ta_check_success("TA_STOCH", retCode)
    return outslowk , outslowd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef STOCHF( np.ndarray high , np.ndarray low , np.ndarray close , int fastk_period, int fastd_period, int fastd_matype):
    """ STOCHF(high, low, close[, fastk_period=?, fastd_period=?, fastd_matype=?])

    Stochastic Fast (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        fastk_period: 5
        fastd_period: 3
        fastd_matype: 0
    Outputs:
        fastk
        fastd
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outfastk
        np.ndarray outfastd
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_STOCHF_Lookback( fastk_period , fastd_period , fastd_matype )
    outfastk = make_double_array(length, lookback)
    outfastd = make_double_array(length, lookback)
    retCode = lib.TA_STOCHF( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , <double *>(outfastk.data)+lookback , <double *>(outfastd.data)+lookback )
    _ta_check_success("TA_STOCHF", retCode)
    return outfastk , outfastd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef STOCHRSI( np.ndarray real , int timeperiod, int fastk_period, int fastd_period, int fastd_matype):
    """ STOCHRSI(real[, timeperiod=?, fastk_period=?, fastd_period=?, fastd_matype=?])

    Stochastic Relative Strength Index (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
        fastk_period: 5
        fastd_period: 3
        fastd_matype: 0
    Outputs:
        fastk
        fastd
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outfastk
        np.ndarray outfastd
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_STOCHRSI_Lookback( timeperiod , fastk_period , fastd_period , fastd_matype )
    outfastk = make_double_array(length, lookback)
    outfastd = make_double_array(length, lookback)
    retCode = lib.TA_STOCHRSI( 0 , endidx , <double *>(real.data)+begidx , timeperiod , fastk_period , fastd_period , fastd_matype , &outbegidx , &outnbelement , <double *>(outfastk.data)+lookback , <double *>(outfastd.data)+lookback )
    _ta_check_success("TA_STOCHRSI", retCode)
    return outfastk , outfastd 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef SUB( np.ndarray real0 , np.ndarray real1 ):
    """ SUB(real0, real1)

    Vector Arithmetic Subtraction (Math Operators)

    Inputs:
        real0: (any ndarray)
        real1: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real0 = check_array(real0)
    real1 = check_array(real1)
    length = check_length2(real0, real1)
    begidx = check_begidx2(length, <double*>(real0.data), <double*>(real1.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SUB_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SUB( 0 , endidx , <double *>(real0.data)+begidx , <double *>(real1.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SUB", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef SUM( np.ndarray real , int timeperiod):
    """ SUM(real[, timeperiod=?])

    Summation (Math Operators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_SUM_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_SUM( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_SUM", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef T3( np.ndarray real , int timeperiod, double vfactor):
    """ T3(real[, timeperiod=?, vfactor=?])

    Triple Exponential Moving Average (T3) (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 5
        vfactor: 0.7
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_T3_Lookback( timeperiod , vfactor )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_T3( 0 , endidx , <double *>(real.data)+begidx , timeperiod , vfactor , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_T3", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef TAN( np.ndarray real ):
    """ TAN(real)

    Vector Trigonometric Tan (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TAN_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TAN( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TAN", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef TANH( np.ndarray real ):
    """ TANH(real)

    Vector Trigonometric Tanh (Math Transform)

    Inputs:
        real: (any ndarray)
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TANH_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TANH( 0 , endidx , <double *>(real.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TANH", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef TEMA( np.ndarray real , int timeperiod):
    """ TEMA(real[, timeperiod=?])

    Triple Exponential Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TEMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TEMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TEMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef TRANGE( np.ndarray high , np.ndarray low , np.ndarray close ):
    """ TRANGE(high, low, close)

    True Range (Volatility Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TRANGE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TRANGE( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TRANGE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef TRIMA( np.ndarray real , int timeperiod):
    """ TRIMA(real[, timeperiod=?])

    Triangular Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TRIMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TRIMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TRIMA", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef TRIX( np.ndarray real , int timeperiod):
    """ TRIX(real[, timeperiod=?])

    1-day Rate-Of-Change (ROC) of a Triple Smooth EMA (Momentum Indicators)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TRIX_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TRIX( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TRIX", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef TSF( np.ndarray real , int timeperiod):
    """ TSF(real[, timeperiod=?])

    Time Series Forecast (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TSF_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TSF( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TSF", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef TYPPRICE( np.ndarray high , np.ndarray low , np.ndarray close ):
    """ TYPPRICE(high, low, close)

    Typical Price (Price Transform)

    Inputs:
        prices: ['high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_TYPPRICE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_TYPPRICE( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_TYPPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef ULTOSC( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod1, int timeperiod2, int timeperiod3):
    """ ULTOSC(high, low, close[, timeperiod1=?, timeperiod2=?, timeperiod3=?])

    Ultimate Oscillator (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod1: 7
        timeperiod2: 14
        timeperiod3: 28
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_ULTOSC_Lookback( timeperiod1 , timeperiod2 , timeperiod3 )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_ULTOSC( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod1 , timeperiod2 , timeperiod3 , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_ULTOSC", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef VAR( np.ndarray real , int timeperiod, double nbdev):
    """ VAR(real[, timeperiod=?, nbdev=?])

    Variance (Statistic Functions)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 5
        nbdev: 1.0
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_VAR_Lookback( timeperiod , nbdev )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_VAR( 0 , endidx , <double *>(real.data)+begidx , timeperiod , nbdev , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_VAR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef WCLPRICE( np.ndarray high , np.ndarray low , np.ndarray close ):
    """ WCLPRICE(high, low, close)

    Weighted Close Price (Price Transform)

    Inputs:
        prices: ['high', 'low', 'close']
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_WCLPRICE_Lookback( )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_WCLPRICE( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_WCLPRICE", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef WILLR( np.ndarray high , np.ndarray low , np.ndarray close , int timeperiod):
    """ WILLR(high, low, close[, timeperiod=?])

    Williams' %R (Momentum Indicators)

    Inputs:
        prices: ['high', 'low', 'close']
    Parameters:
        timeperiod: 14
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    high = check_array(high)
    low = check_array(low)
    close = check_array(close)
    length = check_length3(high, low, close)
    begidx = check_begidx3(length, <double*>(high.data), <double*>(low.data), <double*>(close.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_WILLR_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_WILLR( 0 , endidx , <double *>(high.data)+begidx , <double *>(low.data)+begidx , <double *>(close.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_WILLR", retCode)
    return outreal 

@wraparound(False)  # turn off relative indexing from end of lists
@boundscheck(False) # turn off bounds-checking for entire function
cdef WMA( np.ndarray real , int timeperiod):
    """ WMA(real[, timeperiod=?])

    Weighted Moving Average (Overlap Studies)

    Inputs:
        real: (any ndarray)
    Parameters:
        timeperiod: 30
    Outputs:
        real
    """
    cdef:
        np.npy_intp length
        int begidx, endidx, lookback
        TA_RetCode retCode
        int outbegidx
        int outnbelement
        np.ndarray outreal
    real = check_array(real)
    length = real.shape[0]
    begidx = check_begidx1(length, <double*>(real.data))
    endidx = <int>length - begidx - 1
    lookback = begidx + lib.TA_WMA_Lookback( timeperiod )
    outreal = make_double_array(length, lookback)
    retCode = lib.TA_WMA( 0 , endidx , <double *>(real.data)+begidx , timeperiod , &outbegidx , &outnbelement , <double *>(outreal.data)+lookback )
    _ta_check_success("TA_WMA", retCode)
    return outreal 