cimport ta_formula._indicators as ta
cimport numpy as np

# define datas intervals
datas = [['1m']]

# define constant params
kdj_minvalue = 10
kdj_maxvalue = 90

# define datas params
CLOSE = datas[0][0]['close']
HIGH = datas[0][0]['high']
LOW = datas[0][0]['low']

# define indicators
ma5 = ta.SMA(CLOSE, 5) # 这里只是测试，实际应该考虑用stream_SMA
ma250 = ta.SMA(CLOSE, 250)
skd = ta.stream_SLOW_KD(HIGH, LOW, CLOSE, 69, 3)

# define signals
ret = {
    'open_long_condition1': ta.kup(ma250,-1) and ta.crossdown(ma5, ma250, -1),
    'open_short_condition1': ta.kdown(ma250,-1) and ta.crossup(ma5, ma250, -1),
    'open_long_condition2': skd[0] <= kdj_minvalue and CLOSE[-1] > ma250[-1],
    'open_short_condition2': skd[0] >= kdj_maxvalue and CLOSE[-1] < ma250[-1],
    'close_long': skd[0] >= kdj_maxvalue,
    'close_short': skd[0] <= kdj_minvalue,
    'last_close_price': CLOSE[-1],
    'last_ma250': ma250[-1],
}