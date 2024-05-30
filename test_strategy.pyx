cimport ta_formula._indicators as ta

datas = [['1m']]
kdj_minvalue = 10
kdj_maxvalue = 90

CLOSE = datas[0][0]['close']
HIGH = datas[0][0]['high']
LOW = datas[0][0]['low']

ret = {
    'open_long_condition1': ta.kup(ta.SMA(CLOSE, 250),-1),
    'open_short_condition1': ta.kdown(ta.SMA(CLOSE, 250),-1),
    'open_long_condition2': ta.stream_SLOW_KD(HIGH, LOW, CLOSE, 69, 3)[0] <= kdj_minvalue,
    'open_short_condition2': ta.stream_SLOW_KD(HIGH, LOW, CLOSE, 69, 3)[0] >= kdj_maxvalue,
    'close_long': ta.stream_SLOW_KD(HIGH, LOW, CLOSE, 69, 3)[0] >= kdj_maxvalue,
    'close_short': ta.stream_SLOW_KD(HIGH, LOW, CLOSE, 69, 3)[0] <= kdj_minvalue,
}