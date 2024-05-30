from ._ta_lib cimport TA_RetCode
from . cimport _ta_lib as lib

cdef void _ta_check_success(str function_name, TA_RetCode ret_code):
    cdef str description
    if ret_code == 0:
        return
    elif ret_code == 1:
        description = 'Library Not Initialized (TA_LIB_NOT_INITIALIZE)'
    elif ret_code == 2:
        description = 'Bad Parameter (TA_BAD_PARAM)'
    elif ret_code == 3:
        description = 'Allocation Error (TA_ALLOC_ERR)'
    elif ret_code == 4:
        description = 'Group Not Found (TA_GROUP_NOT_FOUND)'
    elif ret_code == 5:
        description = 'Function Not Found (TA_FUNC_NOT_FOUND)'
    elif ret_code == 6:
        description = 'Invalid Handle (TA_INVALID_HANDLE)'
    elif ret_code == 7:
        description = 'Invalid Parameter Holder (TA_INVALID_PARAM_HOLDER)'
    elif ret_code == 8:
        description = 'Invalid Parameter Holder Type (TA_INVALID_PARAM_HOLDER_TYPE)'
    elif ret_code == 9:
        description = 'Invalid Parameter Function (TA_INVALID_PARAM_FUNCTION)'
    elif ret_code == 10:
        description = 'Input Not All Initialized (TA_INPUT_NOT_ALL_INITIALIZE)'
    elif ret_code == 11:
        description = 'Output Not All Initialized (TA_OUTPUT_NOT_ALL_INITIALIZE)'
    elif ret_code == 12:
        description = 'Out-of-Range Start Index (TA_OUT_OF_RANGE_START_INDEX)'
    elif ret_code == 13:
        description = 'Out-of-Range End Index (TA_OUT_OF_RANGE_END_INDEX)'
    elif ret_code == 14:
        description = 'Invalid List Type (TA_INVALID_LIST_TYPE)'
    elif ret_code == 15:
        description = 'Bad Object (TA_BAD_OBJECT)'
    elif ret_code == 16:
        description = 'Not Supported (TA_NOT_SUPPORTED)'
    elif ret_code == 5000:
        description = 'Internal Error (TA_INTERNAL_ERROR)'
    elif ret_code == 65535:
        description = 'Unknown Error (TA_UNKNOWN_ERR)'
    else:
        description = 'Unknown Error'
    raise Exception('%s function failed with error code %s: %s' % (
        function_name, ret_code, description))

cdef enum FUNC_UNST_IDS:
    ID_ADX, ID_ADXR, ID_ATR, ID_CMO, ID_DX, ID_EMA, ID_HT_DCPERIOD, ID_HT_DCPHASE, ID_HT_PHASOR, \
    ID_HT_SINE, ID_HT_TRENDLINE, ID_HT_TRENDMODE, ID_KAMA, ID_MAMA, ID_MFI, ID_MINUS_DI, \
    ID_MINUS_DM, ID_NATR, ID_PLUS_DI, ID_PLUS_DM, ID_RSI, ID_STOCHRSI, ID_T3, ID_ALL

cdef void _ta_set_unstable_period(FUNC_UNST_IDS func_unst_id, unsigned int period):
    cdef TA_RetCode ret_code
    ret_code = lib.TA_SetUnstablePeriod(func_unst_id, period)
    _ta_check_success('TA_SetUnstablePeriod', ret_code)

cdef unsigned int _ta_get_unstable_period(FUNC_UNST_IDS func_unst_id):
    cdef unsigned int period
    period = lib.TA_GetUnstablePeriod(func_unst_id)
    return period

cdef void _ta_set_compatibility(int value):
    cdef TA_RetCode ret_code
    ret_code = lib.TA_SetCompatibility(value)
    _ta_check_success('TA_SetCompatibility', ret_code)

cdef int _ta_get_compatibility():
    cdef int value
    value = lib.TA_GetCompatibility()
    return value
