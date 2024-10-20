#cython: language_level=3str

# Usage:
# 1. build: change numpy include path to your own path
#   Linux/MacOS: CPPFLAGS="-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION" C_INCLUDE_PATH=/Users/qiong/opt/miniconda3/lib/python3.9/site-packages/numpy/core/include cythonize -i --3str tests/indicators_cases.pyx
#   Windows: 在系统环境变量中添加INCLUDE，添加numpy的include和ta-lib的include，然后打开cmd执行cythonize -i --3str tests/indicators_cases.pyx
# 2. run test: python -m pytest tests

from ta_formula cimport indicators as ta
import numpy as np

cdef double NaN = np.nan

cdef extern from "math.h":
    bint isnan(double x)

def np_nan():
    cdef double a = NaN
    cdef double b = 0
    assert NaN != NaN
    assert a != a
    assert a != NaN
    assert b != NaN
    assert b == b
    assert isnan(a)
    assert not isnan(b)

def np_ext():
    cdef int[::1] a = np.array([1, 2, 4, 5], dtype=np.intc)
    b = ta.rolling_sum(a, 1)
    np.testing.assert_array_equal(b, [1,2,4,5])
    b = ta.rolling_sum(a, 2)
    np.testing.assert_array_equal(b, [NaN,3,6,9])
    b = ta.rolling_sum(a, 3)
    np.testing.assert_array_equal(b, [NaN,NaN,7,11])
    b = ta.rolling_sum(a, 4)
    np.testing.assert_array_equal(b, [NaN,NaN,NaN,12])
    b = ta.rolling_sum(a, 5)
    np.testing.assert_array_equal(b, [NaN,NaN,NaN,NaN])
    cdef double[::1] a1 = np.array([NaN, 2.0, 4.0, 5.0], dtype=np.double)
    b = ta.rolling_sum(a1, 2)
    np.testing.assert_array_equal(b, [NaN,NaN,6,9])
    cdef double[::1] a12 = np.array([NaN, NaN, 4.0, 5.0], dtype=np.double)
    b = ta.rolling_sum(a12, 3)
    np.testing.assert_array_equal(b, [NaN,NaN,NaN,NaN])
    cdef double[::1] a11 = np.array([1, 2.0, 4.0, NaN], dtype=np.double)
    b = ta.rolling_sum(a11, 2)
    np.testing.assert_array_equal(b, [NaN,3,6,NaN])
    cdef double[::1] a111 = np.array([1, 2.0, NaN, 3, 4.0, 5.0], dtype=np.double)
    b = ta.rolling_sum(a111, 3)
    np.testing.assert_array_equal(b, [NaN,NaN,NaN,NaN,NaN,12])
    cdef long long[::1] a2 = np.array([1, -2, 3, -4], dtype=np.longlong)
    b = ta.rolling_sum(a2, 3)
    np.testing.assert_array_equal(b, [NaN,NaN,2,-3])

    a3 = np.array([1, NaN, 4, 5], dtype=np.double)
    ta.ffill(a3)
    np.testing.assert_array_equal(a3, [1,1,4,5])

    a4 = np.array([1, 0, 4, 0], dtype=np.double)
    ta.replace(a4, 0, NaN)
    np.testing.assert_array_equal(a4, [1, NaN, 4, NaN])

    ta.shift_inplace(a4, 1)
    np.testing.assert_array_equal(a4, [NaN, 1, NaN, 4])
    ta.shift_inplace(a4, -1)
    np.testing.assert_array_equal(a4, [1, NaN, 4, NaN])

    a4 = ta.shift(a4, 4)
    np.testing.assert_array_equal(a4, [NaN, NaN, NaN, NaN])


def bias(data):
    bias = ta.BIAS(data['close'], 20)
    np.testing.assert_array_almost_equal(bias[-3:], [0.18013025, 0.15239487, 0.13410818])
    cdef double sbias = ta.stream_BIAS(data['close'], 20)
    assert sbias == 0.13410818059901194
    rbias = ta.recent_BIAS(data['close'], 20, 3)
    np.testing.assert_array_almost_equal(rbias, [0.18013025, 0.15239487, 0.13410818])

def macd(data):
    dif, dea, hist = ta.MACD(data['close'], 12, 26, 9)
    np.testing.assert_array_almost_equal(dif[-3:], [14.78609958, 14.20392231, 13.58593246])
    np.testing.assert_array_almost_equal(dea[-3:], [14.81383313, 14.69185097, 14.47066726])
    np.testing.assert_array_almost_equal(hist[-3:], [-0.0554671,  -0.97585731, -1.7694696 ])

    cdef ta.tuple_double3 macd = ta.stream_MACD(data['close'], 12, 26, 9)
    sdif, sdea, shist = macd
    assert sdif == 13.585613546212699
    assert sdea == 14.470194405515944
    assert shist == -1.7691617186064903

    rdif, rdea, rhist = ta.recent_MACD(data['close'], 12, 26, 9, 3)
    np.testing.assert_array_almost_equal(rdif[-3:], [14.78611383, 14.20393551, 13.58594468])
    np.testing.assert_array_almost_equal(rdea[-3:], [14.81385249, 14.69186909, 14.47068421])
    np.testing.assert_array_almost_equal(rhist[-3:], [-0.05547732, -0.97586717, -1.76947906])

def kd(data):
    k, d = ta.KD(data['high'], data['low'], data['close'], 9, 3, 3)
    np.testing.assert_array_almost_equal(k[-3:], [67.704544,   63.65488118, 57.58810261])
    np.testing.assert_array_almost_equal(d[-3:], [70.76549187, 68.39528831, 64.79289308])
    cdef ta.tuple_double2 kd = ta.stream_KD(data['high'], data['low'], data['close'], 9, 3, 3)
    sk, sd = kd
    assert sk == 57.58810260608417
    assert sd == 64.79291116270358
    rk, rd = ta.recent_KD(data['high'], data['low'], data['close'], 9, 3, 3, 3)
    np.testing.assert_array_almost_equal(rk[-3:], [67.704544,   63.65488118, 57.58810261])
    np.testing.assert_array_almost_equal(rd[-3:], [70.76549573, 68.39529088, 64.79289479])

def kdj(data):
    k, d, j = ta.KDJ(data['high'], data['low'], data['close'], 9, 3, 3)
    print('vector:', k[-3:], d[-3:], j[-3:])
    np.testing.assert_array_almost_equal(k[-3:], [67.704544,   63.65488118, 57.58810261])
    np.testing.assert_array_almost_equal(d[-3:], [70.76549187, 68.39528831, 64.79289308])
    np.testing.assert_array_almost_equal(j[-3:], [61.58264824, 54.17406693, 43.17852167])
    cdef ta.tuple_double3 kdj = ta.stream_KDJ(data['high'], data['low'], data['close'], 9, 3, 3)
    sk, sd, sj = kdj
    assert sk == 57.58810260608417
    assert sd == 64.79291116270358
    assert sj == 43.17848549284537
    rk, rd, rj = ta.recent_KDJ(data['high'], data['low'], data['close'], 9, 3, 3, 3)
    np.testing.assert_array_almost_equal(rk, [67.704544,   63.65488118, 57.58810261])
    np.testing.assert_array_almost_equal(rd, [70.76549573, 68.39529088, 64.79289479])
    np.testing.assert_array_almost_equal(rj, [61.58264053, 54.17406179, 43.17851824])

def skd(data):
    k, d = ta.SLOW_KD(data['high'], data['low'], data['close'], 69, 3)
    assert k[-100] == 15.059984640538595
    assert d[-100] == 12.902647213651875
    cdef ta.tuple_double2 skd = ta.stream_SLOW_KD(data['high'], data['low'], data['close'], 69, 3)
    ssk, ssd = skd
    assert ssk == 89.97947536213022
    assert ssd == 90.12987900350545
    rsk, rsd = ta.recent_SLOW_KD(data['high'], data['low'], data['close'], 69, 3, 100)
    assert rsk[0] == 15.059984640538595
    assert rsd[0] == 12.902647213651889

def amplitude(data):
    amp = ta.AMPLITUDE(data['high'], data['low'], data['close'], 5)
    np.testing.assert_array_almost_equal(amp[-5:], [0.00075538, 0.00125707, 0.00125818, 0.00062838, 0.0])
    cdef double samp = ta.stream_AMPLITUDE(data['high'], data['low'], data['close'], 5)
    assert samp == 0.0
    ramp = ta.recent_AMPLITUDE(data['high'], data['low'], data['close'], 5, 5)
    np.testing.assert_array_almost_equal(ramp, [0.00075538, 0.00125707, 0.00125818, 0.00062838, 0.0])

def zig(data):
    points = ta.ZIG(data['close'], 0.002)
    assert points[-1] == 7952.0
