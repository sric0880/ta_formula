# Matype

## talib matype

- TA_MAType TA_MAType_SMA = 0

  Simple Moving Average

  等于看盘软件里的MA

- TA_MAType TA_MAType_EMA = 1

  Exponential Moving Average 指数移动平均

  等于看盘软件的EMA

  ```c
  EMA(n) = Price * 2/(n+1) + PreEMA * (n-1)/(n+1)
  ```

- TA_MAType TA_MAType_WMA = 2
  
  Weighted Moving Aeverage  加权移动平均
  
  等于看盘软件的WMA
  
  ```c
  WMA(3) = (x1*1)+(x2*2)+(x3*3) / 6
  ```

- TA_MAType TA_MAType_DEMA = 3

  Double EMA
  
  ```c
  EMA2 = EMA(EMA(t,period),period)
  DEMA = 2*EMA(t,period)- EMA2
  ```

- TA_MAType TA_MAType_TEMA = 4

  Triple EMA
  
  ```c
  EMA1 = EMA(t,period)
  EMA2 = EMA(EMA(t,period),period)
  EMA3 = EMA(EMA(EMA(t,period),period))
  TEMA = 3*EMA1 - 3*EMA2 + EMA3
  ```

- TA_MAType TA_MAType_TRIMA = 5

  triangular MA
  
  类似WMA，WMA将更多权重放在最近，TRIMA将更多权重放在中间
  
  ```c
  TRIMA(5) = ((1*a)+(2*b)+(3*c)+(2*d)+(1*e)) / 9
  ```

- TA_MAType TA_MAType_KAMA = 6

  Kaufman Adaptive Moving Average
  
  ```c
  KAMA(10,2,30):
  
  Change = ABS(Close - Close (10 periods ago))
  Volatility = Sum10(ABS(Close - Prior Close))
  ER = Change/Volatility
  SC = [ER x (2/(2+1) - 2/(30+1)) + 2/(30+1)]^2
  Current KAMA = Prior KAMA + SC x (Price - Prior KAMA)
  ```

- TA_MAType TA_MAType_MAMA = 7
  MESA Adaptive Moving Average
  
  ```c
  TODO:
  ```

- TA_MAType TA_MAType_T3 = 8
  
  Triple Exponential Moving Average

  ```c
  T3(Period, vFactor):

  EMA1(x,Period) = EMA(x,Period)
  EMA2(x,Period) = EMA(EMA1(x,Period),Period)
  GD(x,Period,vFactor) = (EMA1(x,Period)*(1+vFactor)) - (EMA2(x,Period)*vFactor)
  T3 = GD (GD ( GD(t, Period, vFactor), Period, vFactor), Period, vFactor);
  ```

## 看盘软件

看盘软件中特有的平均公式(全是围绕SMA做文章)

- SMA(X,N,M)
  
  ```c
  Y=(X*M+Y'*(N-M))/N
  ```

  当n=2*N-1时，可以和EMA(n)互换，所以talib中没有SMA

- DMA(X,A)
  
  ```c
  Y=A*X+(1-A)*Y'
  ```

  和 SMA一样，只是用小数表达参数而已

- MEMA(X,N)
  
  ```c
  Y=(X+Y'*(N-1))/N
  ```
  
  和 SMA(X,N,1)、MA(X,N)一样，毫无意义的公式
  
- TMA(X,N,M)
  
  ```c
  Y=(N*Y'+M*X)
  ```
  
  如果N+M==1, 那么和DMA一样
