cimport ta_formula._indicators as ta

# define params
datas = [['1m']]
kdj_minvalue = 10
kdj_maxvalue = 90

# define datas
CLOSE = datas[0][0]['close']
HIGH = datas[0][0]['high']
LOW = datas[0][0]['low']

# define indicators
ma250 = ta.SMA(CLOSE, 250)
skd = ta.stream_SLOW_KD(HIGH, LOW, CLOSE, 69, 3)

# define signals
ret = {
    'open_long_condition1': ta.kup(ma250,-1),
    'open_short_condition1': ta.kdown(ma250,-1),
    'open_long_condition2': skd[0] <= kdj_minvalue,
    'open_short_condition2': skd[0] >= kdj_maxvalue,
    'close_long': skd[0] >= kdj_maxvalue,
    'close_short': skd[0] <= kdj_minvalue,
}