# ta_formula

[![PyPI](https://github.com/sric0880/ta_formula/actions/workflows/python-publish.yml/badge.svg)](https://github.com/sric0880/ta_formula/actions/workflows/python-publish.yml) [![PyPI](https://img.shields.io/pypi/v/ta_formula)](https://pypi.org/project/ta-formula/)

## 依赖库

- [ta-lib底层lib库](https://anaconda.org/conda-forge/libta-lib/files)==0.4.0

## 安装使用

```sh
pip install ta_formula
```

参考案例 [examples.ipynb](https://github.com/sric0880/ta_formula/blob/main/examples.ipynb)

- [matype.md](https://github.com/sric0880/ta_formula/blob/main/matype.md) 所有移动平均算法参考
- [cdl_pattern_recognition.md](https://github.com/sric0880/ta_formula/blob/main/cdl_pattern_recognition.md) 所有TA-Lib中的K线形态识别算法参考

## 特性

- 自定义指标，在[TA-Lib(0.4.29)](https://pypi.org/project/TA-Lib/)库([Github](https://github.com/TA-Lib/ta-lib-python))的基础上扩展自己的指标，完全在Cython中实现。
- 自定义区间指标。
- 自定义策略（同样纯Cython实现），根据参数和输入数据即时编译成动态链接库，实现微秒级信号发现
  - 普通台式机CPU i5-10400 @2.9GHz 大概一个指标计算在0.5~5微秒（timeit结果，实际运行考虑到缓存缺失，会慢很多，可以尝试多进程方案）
  - 支持传入不同市场多个标的物
  - 支持传参
  - 支持自定义返回字段
  - 一次计算中，相同指标计算结果缓存，不重复计算
- 数据流入、信号流出框架，自定义数据流，数据流支持asyncio，多线程，相同策略相同数据去重，避免重复计算。

## 策略文件示例

```py
cimport ta_formula.indicators as ta
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
ma5 = ta.SMA(CLOSE, 5)
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
```

见 [test_strategy.pyx](https://github.com/sric0880/ta_formula/blob/main/test_strategy.pyx)

返回示例：

```json
{
  "symbols": [["shanghai001", "ag2412"]],
  "data_update_time": 0,
  "data_rec_time": 17174862888170903,
  "calc_time": 269900,
  "open_long_condition1": false,
  "open_short_condition1": false,
  "open_long_condition2": false,
  "open_short_condition2": false,
  "close_long": false,
  "close_short": false,
  "last_close_price": 7952.0,
  "last_ma250": 7922.2
}
```

其中`symbols`, `data_update_time`, `data_rec_time`、`calc_time`为附加返回字段，分别表示:

- symbols返回当前策略计算用的金融标的组合
- data_update_time数据更新的时间，由数据服务器发送过来，单位以数据后台为准
- 接收到数据的时间戳，单位微妙，int，系统时间可能有误差
- 策略从接收数据，到计算完成，发送信号经过的时间，单位纳秒，int

通过比较三者的时间差，可以大致知道计算延迟和网络延迟

其他字段为自定义返回字段

## TODO

1. 所有`stream_XXX`指标函数，需要显式注明返回类型，比如int, double，或者tuple类型，比如(double, double)。如果不标明，返回的不是c类型，而是python类型，比如int返回的是PyInt。目前只有部分函数修改了。Cython不支持python对象的tuple，比如(np.ndarray, np.ndarray)。
2. 将所有ndarray传参修改为double[::1]，可以避免动态类型检查，避免调用Python的引用计数，提高效率。目前只有部分函数改了。
3. ZIG、PERIOD_MAX_BIAS 没有stream和recent函数
4. 多进程数据后台支持

## 指标

一个指标有三个版本，比如MACD：

1. `MACD`: 从头到尾计算所有指标，返回ndarray。
2. `stream_MACD`: 只计算最后一天的指标，返回double,或者tuple(double,double)等。
3. `recent_MACD`: 计算最近`calc_length`天的指标，返回ndarray，当`calc_length==1`时，效果和`stream_MACD`一样。

## 已扩展的方法

指标含义及用法见代码`indicators.pyx`注释

```c
// 一般指标
SMA, BIAS, MACD, STOCH, KD, KDJ, SLOW_KD AMPLITUDE, ZIG,

// 区间指标
PERIOD_MAX_BIAS

// numpy 函数
shift_inplace shift replace ffill rolling_sum
```

## 扩展TA-Lib

`_ta_lib_xxx`文件是从TA-Lib源码复制过来的。安装完TA-Lib之后，不会安装对应的pxd和pxi文件，所以这里的pxd和pxi直接从TA-Lib源码复制过来。

复制过来的方法，是`def`定义的，全部改成了`cdef`，只允许c内部调用。如果要用python测试，可以封装成`strategy`。

如果要改写TA-Lib的方法，`_func.pxi`源文件、或`_stream.pxi`源文件已经复制到对应`_ta_lib_xxx.pxi`，直接在里面改写就行。但是要添加好注释。

添加`recent_xxx`方法：

从TA-Lib`_stream.pxi`源文件直接复制到`indicators.pyx`改写，并改名为`recent_xxx`

## stream_xxx 和 recent_xxx 函数计算精度的问题

使用任何talib库中的stream函数时，都要测试他和非stream函数的返回是否一致

talib计算stream指标时，只计算最后一天的值，但是会往前查看历史数据，一般长度为timeperiod+不稳定期限。

比如EMA(3), 3天前的数据也会影响最后结果，不稳定期限越长，最后结果越精确。

talib所有指标默认不稳定期限为0，要满足自己的精度要求，需要自己设置不稳定期限。

不同的指标，不同的timeperiod，需要设置不同的不确定长度，才能达到相同的精度。

设置不确定长度的方法为：__ta_set_unstable_period(非线程安全)

talib计算精度受历史数据长度影响的指标有:

```c
ADX, ADXR, ATR, CMO, DX, EMA, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR,
HT_SINE, HT_TRENDLINE, HT_TRENDMODE, KAMA, MAMA, MFI, MINUS_DI,
MINUS_DM, NATR, PLUS_DI, PLUS_DM, RSI, STOCHRSI, T3
```
