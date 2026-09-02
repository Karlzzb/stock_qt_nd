# 社区特征收割：股票短期（5-30 交易日）收益预测特征全集

> 收割日期：2026-09-01。
> 场景：A 股事件驱动狙击模型，事件 = MACD 底背离信号日 T，T+1 开盘买入，在信号池内做排序二次筛选。
> 数据约束：仅日线 OHLCV + 成交额（可派生 vwap = amount / volume）+ 换手率（volume / 流通股本，可选）。
> 硬约束：所有特征必须"信号日 T 收盘后可算"，即只使用 ≤ T 日的数据。

## 0. 结论速览

- 共收割 8 个来源、约 1300+ 个具名特征/指标（去重前），可归纳为约 150 个时序/截面算子 × 若干窗口参数的组合生成体系。
- 对 5-30 日持仓最直接可用的三大成体系来源：qlib Alpha158（158 个，语法完全程序化）、国泰君安 Alpha191（191 个，日频价量，专为短周期设计）、TA-Lib/通达信经典指标（约 100 个非形态类）。
- WorldQuant Alpha101 与 Alpha158 高度同构，其价值在于提供了 RANK/CORR/DECAY_LINEAR 等算子组合的"配方范式"。
- 本文第 13 节给出与本场景（底背离池内排序）最匹配的 Top 20 特征。
- 全部列出特征均满足"T 收盘后可算"，唯一需要注意的是 GTJA Alpha191 中少数因子需要指数行情（index_open/index_close）或三因子序列，应剔除或改造。

## 1. 来源覆盖清单

| # | 来源 | 链接 | 拿到的东西 |
|---|---|---|---|
| 1 | 微软 qlib Alpha158 / Alpha360 源码 | https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py | 158 个特征的全部表达式、29 个 rolling 算子、5 个默认窗口；Alpha360 构造语法 |
| 2 | qlib 表达式引擎算子库 | https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py | 46 个算子类全清单（可程序化生成特征） |
| 3 | TA-Lib 官方函数文档 | https://ta-lib.org/functions/ ；镜像 https://github.com/TA-Lib/ta-lib-python/tree/master/docs/func_groups | 10 个家族 160 个函数全清单及签名 |
| 4 | 国泰君安《基于短周期价量特征的多因子选股体系》（2017-06-15，数量化专题之九十三） | https://guorn.com/static/upload/file/3/134065454575605.pdf | Alpha191 的原始研报、算子定义、因子明细表、4 个实证最强代表因子（含 IC/ICIR） |
| 5 | WorldQuant《101 Formulaic Alphas》 | https://arxiv.org/abs/1601.00991 | 101 因子算子集与配方范式 |
| 6 | 聚宽因子库文档 | https://www.joinquant.com/help/api/help#factor_values ；Alpha191 实现 https://www.joinquant.com/help/api/help?name=Alpha191 | 量价/情绪/技术/动量因子代码清单 |
| 7 | BigQuant 因子表达式（bigexpr）与因子库 | https://bigquant.com/wiki/doc/vBx81T1t8I ；https://bigquant.com/doc/data_features.html ；Alpha191 构建公式 https://bigquant.com/wiki/doc/alpha191-Pyf0TYya6H | bigexpr 算子全集、预计算量价因子字段 |
| 8 | 米筐 RQFactor / rqdatac 技术指标因子 | https://www.ricequant.com/doc/rqfactor/api/built-in-operators ；https://www.ricequant.com/doc/rqdata/python/stock-mod | 约 60 个内置算子、约 90 个技术指标因子的精确公式 |
| 9 | MyTT（通达信/同花顺公式 Python 移植） | https://github.com/mpquant/MyTT | 0 级核心函数 + 35 个应用层指标，通达信口径 |
| 10 | GitHub 因子库项目 | 见第 11 节 | qweave、KunQuant、Alpha101 实现等 |

## 2. qlib Alpha158：完整构造语法

来源：`qlib/contrib/data/loader.py` 中 `Alpha158DL.get_feature_config()`（本报告已核对 main 分支源码原文）。
158 = K 线形态 9 + 原始价格 4 + rolling 算子 29 × 窗口 5（[5, 10, 20, 30, 60]）= 145。
默认配置不含原始 volume 组（`"volume"` 键不在默认 config 中）。
所有特征都做了无量纲化：价格类除以当日 `$close`，成交量类除以 `($volume + 1e-12)`，因此可跨股票比较。

### 2.1 K 线形态组（9 个，kbar）

| 名称 | 表达式 | 含义 |
|---|---|---|
| KMID | `($close-$open)/$open` | 实体长度（相对开盘） |
| KLEN | `($high-$low)/$open` | 全日振幅 |
| KMID2 | `($close-$open)/($high-$low+1e-12)` | 实体占振幅比 |
| KUP | `($high-Greater($open,$close))/$open` | 上影长度 |
| KUP2 | `($high-Greater($open,$close))/($high-$low+1e-12)` | 上影占振幅比 |
| KLOW | `(Less($open,$close)-$low)/$open` | 下影长度 |
| KLOW2 | `(Less($open,$close)-$low)/($high-$low+1e-12)` | 下影占振幅比 |
| KSFT | `(2*$close-$high-$low)/$open` | 收盘在全日区间的位置 |
| KSFT2 | `(2*$close-$high-$low)/($high-$low+1e-12)` | 收盘位置（相对振幅） |

其中 `Greater(a,b)=max(a,b)`，`Less(a,b)=min(a,b)`。

### 2.2 原始价格组（4 个，price）

默认 `windows=[0]`、`feature=["OPEN","HIGH","LOW","VWAP"]`。
生成规则：`d==0` 时 `$field/$close`，否则 `Ref($field, d)/$close`，命名如 `OPEN0`、`VWAP0`。
`$vwap` 在 qlib 数据层 = amount/volume，无 vwap 字段时用 `($high+$low+$close)/3` 或 amount/volume 派生。

### 2.3 rolling 组（29 算子 × 5 窗口 = 145 个）

窗口 `d ∈ {5, 10, 20, 30, 60}`，以下为每个算子的表达式模板（`%d` = d），命名 = 前缀 + d：

| 前缀 | 表达式模板 | 含义 |
|---|---|---|
| ROC | `Ref($close,d)/$close` | d 日前收盘/今收盘（反转动量，<1 表示上涨） |
| MA | `Mean($close,d)/$close` | 均线乖离（d 日均线/今收盘） |
| STD | `Std($close,d)/$close` | 价格波动率 |
| BETA | `Slope($close,d)/$close` | 线性趋势斜率（对时间回归） |
| RSQR | `Rsquare($close,d)` | 趋势 R²（线性度） |
| RESI | `Resi($close,d)/$close` | 回归残差（偏离趋势程度） |
| MAX | `Max($high,d)/$close` | d 日最高/今收盘（距前高） |
| MIN | `Min($low,d)/$close` | d 日最低/今收盘（距前低） |
| QTLU | `Quantile($close,d,0.8)/$close` | 80% 分位价/今收盘 |
| QTLD | `Quantile($close,d,0.2)/$close` | 20% 分位价/今收盘 |
| RANK | `Rank($close,d)` | 今收盘在过去 d 日收盘价中的时序分位 |
| RSV | `($close-Min($low,d))/(Max($high,d)-Min($low,d)+1e-12)` | 随机指标 RSV（KDJ 的 R） |
| IMAX | `IdxMax($high,d)/d` | 最高价距今天数/d（Aroon 部件） |
| IMIN | `IdxMin($low,d)/d` | 最低价距今天数/d |
| IMXD | `(IdxMax($high,d)-IdxMin($low,d))/d` | 高点与低点的时间序（负=先高后低） |
| CORR | `Corr($close,Log($volume+1),d)` | 价-量（对数量）相关 |
| CORD | `Corr($close/Ref($close,1),Log($volume/Ref($volume,1)+1),d)` | 收益-量变相关 |
| CNTP | `Mean($close>Ref($close,1),d)` | d 日上涨天数占比 |
| CNTN | `Mean($close<Ref($close,1),d)` | d 日下跌天数占比 |
| CNTD | `Mean($close>Ref($close,1),d)-Mean($close<Ref($close,1),d)` | 涨跌天数差 |
| SUMP | `Sum(Greater($close-Ref($close,1),0),d)/(Sum(Abs($close-Ref($close,1)),d)+1e-12)` | 总涨幅/总涨跌（RSI 变体） |
| SUMN | `Sum(Greater(Ref($close,1)-$close,0),d)/(Sum(Abs($close-Ref($close,1)),d)+1e-12)` | 总跌幅/总涨跌 = 1-SUMP |
| SUMD | 两者之差 | 涨跌不对称度（类 RSI 居中化） |
| VMA | `Mean($volume,d)/($volume+1e-12)` | 均量/今量（>1 表示缩量） |
| VSTD | `Std($volume,d)/($volume+1e-12)` | 量的波动率 |
| WVMA | `Std(Abs($close/Ref($close,1)-1)*$volume,d)/(Mean(...,d)+1e-12)` | 量加权价格波动 |
| VSUMP | `Sum(Greater($volume-Ref($volume,1),0),d)/(Sum(Abs($volume-Ref($volume,1)),d)+1e-12)` | 量增占比 |
| VSUMN | 对称（量减占比） | = 1-VSUMP |
| VSUMD | 两者之差 | 量的 RSI |

### 2.4 qlib 表达式引擎算子全集（ops.py，46 个类）

程序化生成特征时，除 Alpha158 用到的算子外，引擎还提供以下全集（按类别）：

- 元素级：`Abs, Sign, Log, Power, Add, Sub, Mul, Div, Greater, Less, Gt, Ge, Lt, Le, Eq, Ne, And, Or, Not, If, Mask`
- 滚动时序：`Ref, Mean, Sum, Std, Var, Skew, Kurt, Max, Min, Med, Mad, Quantile, Rank, Count, Delta, Slope, Rsquare, Resi, IdxMax, IdxMin, EMA, WMA`
- 双序列滚动：`Corr, Cov`
- 其他：`ChangeInstrument`（换标的后复权基准）、`TResample`（降采样）、`OpsWrapper`
- 表达式语法支持 `$field`、`Ref(x,n)`、算术/比较/逻辑运算任意嵌套，字段含 `$open/$high/$low/$close/$volume/$amount/$vwap/$turnover`（turnover 取决于数据层是否提供）。
- 文档：https://qlib.readthedocs.io/en/latest/component/data.html

## 3. qlib Alpha360：原始价量快照语法

给模型直接吃"最近 60 日价量轨迹"的方案，共 360 = 6 字段 × 60 lag。
构造规则（对 i = 59..1，再加 i=0）：

```
CLOSE{i}  = Ref($close, i)/$close        （i=0 时为 $close/$close = 1）
OPEN{i}   = Ref($open,  i)/$close
HIGH{i}   = Ref($high,  i)/$close
LOW{i}    = Ref($low,   i)/$close
VWAP{i}   = Ref($vwap,  i)/$close
VOLUME{i} = Ref($volume,i)/($volume+1e-12)
```

即全部用当日 close 归一价格轨迹、用当日 volume 归一量能轨迹。
对本项目价值：可作为 CNN/TCN 类模型的原始输入；对 GBDT 排序模型，优先用 Alpha158 这类已加工统计量。

## 4. TA-Lib 全清单（160 个函数，按家族）

来源：TA-Lib 官方文档（ta-lib.org/functions，经 ta-lib-python 仓库 docs/func_groups 核对）。
全部函数输入均为 ≤ T 日的 OHLCV，满足硬约束。
标注 ★ = 对 5-30 日持仓最相关。

### 4.1 Overlap Studies 均线类（18）

`ACCBANDS, BBANDS★, DEMA, EMA★, HT_TRENDLINE, KAMA, MA★, MAMA, MAVP, MIDPOINT, MIDPRICE, SAR★, SAREXT, SMA★, T3, TEMA, TRIMA, WMA`。
用法要点：raw 均线值不可跨股比较，应加工为 `MA_N/close` 或 `(close-MA_N)/MA_N`（乖离率）。

### 4.2 Momentum 动量类（31）

`ADX★, ADXR, APO, AROON★, AROONOSC★, BOP, CCI★, CMO, DX, IMI★, MACD★, MACDEXT, MACDFIX, MFI★, MINUS_DI, MINUS_DM, MOM★, PLUS_DI, PLUS_DM, PPO★, ROC★, ROCP, ROCR, ROCR100, RSI★, STOCH★(KDJ), STOCHF, STOCHRSI★, TRIX★, ULTOSC, WILLR★`。
关键签名：`ADX(high,low,close,14)`；`MACD(close,12,26,9)`；`RSI(close,14)`；`STOCH(high,low,close,9,3,3)`；`MFI(high,low,close,volume,14)`；`CCI(high,low,close,14)`；`MOM(close,10)`；`PPO(close,12,26)`（可跨股比较的 MACD 百分比版）；`AROON(high,low,14)`。

### 4.3 Volume 量能类（3）

`AD★`（Chaikin A/D，`AD= cumsum(((close-low)-(high-close))/(high-low) * volume)`）、`ADOSC★`（AD 的 3/10 EMA 差）、`OBV★`（能量潮）。
用法要点：AD/OBV 是累积量，须差分或标准化后使用（如 `DELTA(OBV,5)/Mean(volume,20)`）。

### 4.4 Volatility 波动类（3）

`ATR★(high,low,close,14)`、`NATR★`（ATR/close，可跨股比较）、`TRANGE`（真实波幅）。

### 4.5 Price Transform（4）

`AVGPRICE=(o+h+l+c)/4`、`MEDPRICE=(h+l)/2`、`TYPPRICE=(h+l+c)/3`、`WCLPRICE=(h+l+2c)/4`。

### 4.6 Cycle 周期类（5）

`HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE`（Hilbert 变换族，不稳定期长，优先级低）。

### 4.7 Statistic 统计类（9）

`BETA★`（与市场基准回归，需指数数据）、`CORREL★`、`LINEARREG, LINEARREG_ANGLE★, LINEARREG_INTERCEPT, LINEARREG_SLOPE★, STDDEV★, TSF, VAR`。

### 4.8 Math Transform / Math Operators（15 + 11）

`ACOS..TANH` 等 15 个数学变换；`ADD, SUB, MULT, DIV, SUM★, MAX★, MIN★, MAXINDEX★, MININDEX★, MINMAX, MINMAXINDEX` 11 个向量算子（SUM/MAX/MIN/MAXINDEX 是构造自定义特征的基件）。

### 4.9 Pattern Recognition K 线形态（61）

`CDLDOJI, CDLHAMMER★, CDLINVERTEDHAMMER★, CDLENGULFING★, CDLMORNINGSTAR★, CDLPIERCING★, CDLHARAMI, CDL3WHITESOLDIERS, CDLDRAGONFLYDOJI★, CDLTAKURI★, CDLMARUBOZU, CDLSPINNINGTOP, CDLSHOOTINGSTAR, CDLHANGINGMAN, ...`（共 61 个 CDL* 函数）。
输出 ∈ {-100, 0, +100}，全部只用 ≤ T 日 K 线，满足硬约束。
对底背离场景，反转型形态（锤头、倒锤、启明、看涨吞没、蜻蜓十字、Takuri 下影线）可作为离散确认特征。

## 5. 国泰君安 Alpha191（短周期价量因子体系）

### 5.1 出处与性质

- 原始研报：国泰君安证券金融工程《基于短周期价量特征的多因子选股体系——数量化专题之九十三》（2017-06-15，李辰/刘富兵等），PDF：https://guorn.com/static/upload/file/3/134065454575605.pdf 。
- 该报告构建了 191 个日频价量 Alpha 因子（表 6 因子明细，Alpha1–Alpha191），输入仅为个股日频 open/high/low/close/volume/vwap（amount = volume × vwap 派生），与本项目数据约束完全吻合。
- qlib Alpha158 的 rolling 部分源码注释即引用该报告（"Some factor ref" 指向同一 PDF）。
- 开源实现：聚宽 `jqlib.alpha191`（https://www.joinquant.com/help/api/help?name=Alpha191 ）、BigQuant Alpha191（https://bigquant.com/wiki/doc/alpha191-Pyf0TYya6H 、数据表 https://bigquant.com/data/datasources/cn_stock_factors_alpha_191 ）、qweave 内置表达式 `gtja_alpha001–gtja_alpha191`（https://github.com/GaomingOrion/qweave/blob/master/docs/gtja_alpha191.md ）。

### 5.2 Alpha191 算子集（公式语法）

- 时序：`DELAY(X,n)`（=Ref）、`DELTA(X,n)=X-DELAY(X,n)`、`SUM(X,n)`、`MEAN(X,n)`、`STD(X,n)`、`MIN/TSMIN(X,n)`、`MAX/TSMAX(X,n)`、`TSRANK(X,n)`（最新值在 n 日内分位）、`COUNT(cond,n)`、`DECAYLINEAR(X,n)`（线性衰减加权均值，权重 1..n）、`SMA(X,n,m)`（中国式递推平滑，系数 m/n）、`WMA(X,n)`（研报自定义 0.9^i 权重）、`REGBETA(X,SEQUENCE(n))`（对时间回归斜率）、`REGRESI`（回归残差）、`CORR(X,Y,n)`、`COVIANCE(X,Y,n)`。
- 截面：`RANK(X)`（当日全市场分位排名）。
- 元素级：`ABS, SIGN, LOG, SEQUENCE(n), 条件 ?: （三元）`。
- 特殊输入：`BANCHMARKINDEXOPEN/CLOSE`（Alpha75 等少量因子用指数行情）；Alpha30 用 mkt/smb/hml 三因子回归——这两类应剔除或改造。

### 5.3 研报实证最强的 4 个代表因子（3.2 节，含检验值）

研报对风格中性化后的因子收益做了 2010-01 至 2017-04 检验：

| 因子 | 精确公式 | 因子年化收益 | IR | IC | ICIR | 解读 |
|---|---|---|---|---|---|---|
| 价量背离 | `-1 * CORR(vwap_{t-d:t}, volume_{t-d:t})` | 5.27% | 3.78 | 0.003 | 2.65 | 短周期内价跌量增/价升量减 → 后期超额概率高 |
| 开盘缺口 | `Open_t / Close_{t-1}` | 9.00% | 4.83 | 0.005 | 2.94 | 当日跳空短期有动量效应 |
| 异常成交量 | `-1 * Volume_t / MEAN(Volume_{t-d:t})` | 8.35% | 2.01 | 0.005 | 1.71 | 异常缩量是阶段底部特征（需剔除涨跌停日的量异常） |
| 量幅背离 | `-1 * CORR(high_{t-d:t}/low_{t-d:t}, volume_{t-d:t})` | 12.52% | 8.39 | 0.007 | 4.26 | 全部因子中稳定性最强 |

注：研报中 d 为短周期窗口（典型 5–20）。
这 4 个因子与"底背离找反弹"的机制高度同构（缩量、价量背离确认底部），是本场景的一等候选。

### 5.4 代表性公式示例（表 6 原文摘录）

```
Alpha1  = (-1 * CORR(RANK(DELTA(LOG(VOLUME),1)), RANK((CLOSE-OPEN)/OPEN), 6))
Alpha5  = (-1 * TSMAX(CORR(TSRANK(VOLUME,5), TSRANK(HIGH,5),5), 3))
Alpha14 = CLOSE - DELAY(CLOSE,5)
Alpha15 = OPEN/DELAY(CLOSE,1) - 1
Alpha31 = (CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100        （= 乖离率 BIAS12）
Alpha53 = COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100            （= qlib CNTP12×100）
Alpha57 = SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)  （= KDJ 的 K）
Alpha66 = (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
Alpha71 = (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
Alpha76 = STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
Alpha80 = (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100     （5 日量变率）
Alpha101= (不在前 91 行，完整 191 式见研报表 6 或 jqlib.alpha191 源码)
```

## 6. WorldQuant Alpha101

- 论文：Kakushadze, "101 Formulaic Alphas"（2016），https://arxiv.org/abs/1601.00991 。
- 性质：101 个横截面选股公式，输入为日频 OHLCV+volume+vwap(+returns, cap)，持有期天然为日~周级，与 5-30 日匹配。
- 算子集（论文附录）：`rank`（截面分位）、`delay, delta, ts_min, ts_max, ts_argmin, ts_argmax, ts_rank, sum, product, stddev, correlation, covariance, decay_linear, scale（使 sum(abs(x))=1）, sign, signedpower, log, power, min, max, indneutralize`。
- 代表公式：
  - `Alpha#1 = rank(ts_argmax(signedpower(where(returns<0, stddev(returns,20), close), 2), 5)) - 0.5`
  - `Alpha#101 = (close - open) / ((high - low) + 0.001)`（当日实体强度，与 qlib KMID2 同源）
- 开源实现：https://github.com/yli188/WorldQuant_alpha101_code 、https://github.com/CK991357/Quant_alpha101_code 、https://github.com/lvlh2/alpha101 、A 股全流程复现 https://github.com/Parsnip77/Multi-factor-Model-for-Stock-Selection 。
- 注意：含 `indneutralize`（行业中性化）和 `cap` 的因子需要额外数据，改造或剔除。

## 7. 聚宽因子库（量价相关部分）

来源：聚宽官方文档"因子库"（https://www.joinquant.com/help/api/help#factor_values ），`jqfactor.get_all_factors()` 可查全表。
以下摘取量价/情绪/技术/动量家族（均只用日线，T 收盘可算）：

- 情绪类：`WVAD`（威廉变异离散量 = (close-open)/(high-low) × volume）、`VOL5/10/20/60/120/240`（N 日平均换手率）、`DAVOL5/10/20`（N 日均换手/120 日均换手）、`VEMA5/10/12/26`、`VMACD`、`VSTD10/20`、`TVMA6/20`、`TVSTD6/20`、`ATR6/14`、`VROC6/12`、`VR`（成交量比率）、`VOSC`、`AR/BR/ARBR`、`PSY`（心理线）、`money_flow_20`。
- 技术类：`boll_up/boll_down`（布林轨/收盘价）、`EMA5, EMAC10/12/20/26/120`（EMA/收盘价）、`MAC5/10/20/60/120`（MA/收盘价）、`MACDC`、`MFI14`。
- 动量类：`aroon_up_25, aroon_down_25, BBIC, bull_power, bear_power, BIAS5/10/20/60, CCI10/15/20, ROC6/12/20/60/120, TRIX5/10, single_day_VPT`。
- 风险类（短持仓可参考）：`Variance20/60/120, Skewness20/60/120, Kurtosis, sharpe_ratio_20/60/120, 最大回撤`。
- 聚宽同时提供 `jqlib.alpha191` 全套 GTJA Alpha191 实现，公式即第 5 节口径。

## 8. BigQuant bigexpr 表达式引擎 + 预计算因子

### 8.1 bigexpr 算子全集（来源：https://bigquant.com/wiki/doc/vBx81T1t8I ）

- 基础：`where/if, sin..tanh, log/log10/log1p, exp/expm1, sqrt, abs, ceil, floor, sign, signedpower, min, max, isnan, clip, all_quantile, all_wbins, all_cbins`。
- 按股票分组的时序函数：`shift/delay(x,d), delta(x,d), correlation(x,y,d), covariance(x,y,d), residual(x,y), intercept(x,y,d), coefficient(x,y,d), sum, product, std/nanstd, mean/nanmean, var, skew, kurt, decay_linear(x,d), ts_min, ts_max, ts_argmax, ts_argmin, ts_rank`。
- 按日期分组的截面函数：`rank(x)`（当日百分比排名）、`scale(x,a)`（使 sum(abs)=a）。
- 分组函数：`group_mean/group_sum/group_rank(key, x)`（需行业等分组键，本项目可不用）。
- TA-Lib 包装：`ta_ma, ta_macd, ta_kdj, ta_bbands, ta_rsi, ta_atr, ta_cci` 及金叉/死叉信号；K 线形态：红三兵、锤头、早晨之星、乌云盖顶等。
- 示例配方：`rank(rank(close_0/close_1) / rank(close_0/close_10))`、`-1*correlation(open_0, volume_0, 10)`。
- 字段命名约定：`close_0` = 当日，`close_i` = i 日前（`_i` 后缀即 delay）。

### 8.2 预计算量价因子（来源：https://bigquant.com/doc/data_features.html ）

- `return_i / rank_return_i`：过去 i 日收益及其截面排名（i 覆盖 0~360 多个周期）。
- `amount_i / rank_amount_i`：i 日前成交额及排名；`avg_amount_i`：过去 i 日平均成交额。
- `swing_volatility_*, volatility_*`：振幅波动率、收益波动率系列。
- `turn_*, volume_ratio_*`：换手率与量比系列。

## 9. 米筐 RQFactor / rqdatac

### 9.1 RQFactor 内置算子（来源：https://www.ricequant.com/doc/rqfactor/api/built-in-operators ）

- 简单算子：`ABS, LOG, EXP, SIGN, SIGNEDPOWER, MIN/FMIN, MAX/FMAX, IF, EQUAL, REF/DELAY(X,n), DELTA(X,n), PCT_CHANGE(X,n)`。
- 均线：`MA/SMA(X,n), EMA(X,n), EMA_CN（=DMA(X,2/(n+1))）, SMA_CN(X,n,m)（中国式 SMA = DMA(X,m/n)）, WMA/DECAY_LINEAR(X,n), DMA(X,c)`。
- 横截面：`RANK, SCALE, DEMEAN, CS_ZSCORE, QUANTILE, TOP, BOTTOM, INDUSTRY_NEUTRALIZE, CS_REGRESSION_RESIDUAL, CS_FILLNA, FIX（锁定某标的值，如算与沪深300的相关）`。
- 时序统计：`AVEDEV, STD/STDDEV, VAR, TS_SKEW, TS_KURT, SLOPE, SUM, PRODUCT, TS_MIN/LLV, TS_MAX/HHV, TS_ARGMIN, TS_ARGMAX, CORR/CORRELATION(A,B,n), COVARIANCE/COV, COUNT, EVERY, TS_RANK, TS_ZSCORE, TS_REGRESSION(Y,X,n), CROSS(A,B)（金叉/死叉判定）`。
- 这套算子与 qlib/BigQuant 高度同构，三家互相印证了"表达式引擎 = 字段 × 算子 × 窗口"的生成范式。

### 9.2 rqdatac 技术指标因子（来源：https://www.ricequant.com/doc/rqdata/python/stock-mod ，含精确公式）

均线类：`MACD_DIFF/DEA/HIST(12,26,9)`、`TRIX/MATRIX(12,20)`、`BOLL/UP/DOWN(20,2)`、`ASI/ASIT`、`MA/EMA/HMA/LMA/VMA/AMV N∈{3,5,10,20,30,55,60,120,250}`、`VOL N`（N 日平均换手率 = MA(100×volume/流通股本, N)）、`DAVOL5/10/20`（= VOL_N/VOL_120）、`BBI/BBIBOLL(3,6,12,24)`、`DPO/MADPO(20,10,6)`、`MCST`（市场成本 = DMA(amount/volume, 换手)）。
超买超卖类：`OBOS(10)`、`KDJ_K/D/J(9,3,3)`、`RSI6/10`、`WR(10,6)`、`LWR1/2(9,3,3)`、`BIAS5/10/20`、`BIAS36/BIAS612/MABIAS`、`ACCER(8)=SLOPE(close,8)/close`、`CYF(21)`、`SWL/SWS`（分水岭）、`ADTM/MAADTM(23,8)`、`TR/ATR(14,9)`、`DKX/MADKX`（多空线，20 日递加权重均线）、`TAPI/MATAPI`、`OSC(10)=100×(close-MA(close,10))`、`CCI(14)`、`ROC(12)`、`MFI(14)`、`MTM/MAMTM(14)`、`MARSI6/10`、`SKD_K/D(9,3)`、`UDL(3,5,10,20)`、`DI1/DI2/ADX/ADXR(DMI, 14,6)`。
能量类：`AR/BR(26)`、`VR/MAVR(26,6)`、`CR/MACR1-4(26,…)`、`MASS/MAMASS(9,25,6)`、`SY(9)`（心理线 = COUNT(close>ref(close,1),9)/9×100）、`PCNT`、`CYR/MACYR(13,5)`（市场强弱，基于 amount/volume 的 EMA 变化率）、`AMP1/3/5/10/20/60`（振幅 = (HHV(high,N)-LLV(low,N))/ref(close,N)）、`WMA N`、`VOLT20/60`（收盘价 std）、`MDD20/60`（最大回撤）、`AROON_UP/DOWN(14)`、`QTYR_5_20`（量比 = MA(vol,5)/MA(vol,20)）、`OBV`。
另有 `WorldQuant_alpha001–101` 预计算因子。

## 10. MyTT / 通达信 / 同花顺 / TradingView 常用指标

来源：MyTT（https://github.com/mpquant/MyTT ，通达信/同花顺公式纯 pandas 移植，结果与行情软件一致到小数点后 2 位）。

- 0 级核心函数（通达信公式语言基件）：`REF, DIFF, STD, SUM, HHV, LLV, HHVBARS, LLVBARS, MA, EMA, SMA, WMA, DMA, AVEDEV, SLOPE, FORCAST, CROSS, LONGCROSS, COUNT, EVERY, EXIST, FILTER, BARSLAST, CONST, LAST, VALUEWHEN`。
  其中 `HHVBARS/LLVBARS`（最高/最低价距今天数）等价 qlib `IdxMax/IdxMin`，是刻画"底背离结构"的关键算子；`BARSLAST`（上次条件成立距今）可直接刻画"距前低/距金叉天数"。
- 1 级指标（通达信口径，适合 5-30 日持仓者重点）：`MACD, KDJ, RSI, BOLL, ATR/TR, CCI, PSY, DMI(PDI/MDI/ADX/ADXR), WR, BIAS, ASI, VR, ARBR/BRAR, DPO, TRIX, DMA/DFMA, MTM, MASS, ROC, EXPMA, OBV, MFI, EMV, CR, XS2(薛斯通道), TOPRANGE/LOWRANGE`。
- TradingView 常用且以上未覆盖的：`VWAP 及锚定 VWAP（日内口径，日线可用 amount/volume 近似）、Supertrend（ATR 通道，趋势过滤）、Chandelier Exit（吊灯止损线，22 日 ATR×3，适合 5-30 日持仓的离场参考）、Donchian Channel（N 日高低通道，= Max(high,N)/Min(low,N)）、CMF（Chaikin Money Flow，20 日 = SUM(ADL日值,20)/SUM(volume,20)）、Keltner Channel（EMA±ATR）、Elder Ray（bull/bear power = high-EMA13 / low-EMA13）、Fisher Transform、Vortex（VI±）、Choppiness Index（震荡度，区分趋势/盘整，背离后接趋势行情时可过滤）`。
- 通达信特色：`量比`（当日每分钟均量/过去 5 日每分钟均量，日线近似 = volume/MA(volume,5)）、`换手率`系列、`AMOUNT` 派生因子。

## 11. GitHub 因子库项目索引

| 项目 | 链接 | 内容 |
|---|---|---|
| microsoft/qlib | https://github.com/microsoft/qlib | Alpha158/360 + 表达式引擎（46 算子） |
| TA-Lib/ta-lib-python | https://github.com/TA-Lib/ta-lib-python | 160 函数及分组文档 |
| GaomingOrion/qweave | https://github.com/GaomingOrion/qweave | GTJA Alpha191 + WorldQuant Alpha101 的内置表达式实现（含口径说明） |
| Menooker/KunQuant | https://github.com/Menooker/KunQuant | 因子表达式编译器，内置 Alpha101/158/191 批量生成与并行计算 |
| yli188/WorldQuant_alpha101_code | https://github.com/yli188/WorldQuant_alpha101_code | Alpha101 公式实现 |
| CK991357/Quant_alpha101_code | https://github.com/CK991357/Quant_alpha101_code | Alpha101 实现 |
| mpquant/MyTT | https://github.com/mpquant/MyTT | 通达信/同花顺公式移植 |
| 聚宽 jqlib.alpha191 | https://www.joinquant.com/help/api/help?name=Alpha191 | GTJA Alpha191 官方复现 |
| quantskills/skill-factor-alpha191-alpha101 | https://github.com/quantskills/skill-factor-alpha191-alpha101 | Alpha101/158/191 公式参考汇总 |
| Parsnip77/Multi-factor-Model-for-Stock-Selection | https://github.com/Parsnip77/Multi-factor-Model-for-Stock-Selection | A 股日线 Alpha101 复现 + 因子评估全流程 |

## 12. 硬约束核对（"信号日 T 收盘后可算"）

| 来源/家族 | 所需字段 | T 收盘可算？ | 备注 |
|---|---|---|---|
| qlib Alpha158 全部 | OHLCV + vwap(可派生) | 成立 | 表达式中 Ref/rolling 均只看历史 |
| qlib Alpha360 全部 | OHLCV + vwap | 成立 | |
| TA-Lib 全部 160 | OHLCV(+volume) | 成立 | BETA 需基准指数序列；形态类只看 ≤T |
| GTJA Alpha191 主体 | OHLCV + vwap | 成立 | 例外：Alpha75 等用指数 OHLC（需指数数据，可补）；Alpha30 用 mkt/smb/hml（剔除） |
| WorldQuant Alpha101 主体 | OHLCV + vwap | 成立 | 例外：含 cap/indneutralize 的因子需市值/行业（剔除或改造）；截面 rank 在信号池内当日可算 |
| 聚宽量价/情绪/技术/动量 | OHLCV + 换手率 | 成立 | VOL*/DAVOL* 需流通股本（换手率可由 volume/流通股本派生） |
| BigQuant bigexpr 时序类 | OHLCV + amount | 成立 | group_* 需分组键；rank_return_i 为截面口径 |
| 米筐技术指标因子 | OHLCV + amount + 换手率 | 成立 | VOL/DAVOL/MCST/CYF 需流通股本 |
| MyTT/通达信/TradingView | OHLCV(+amount) | 成立 | 日内 VWAP 用 amount/volume 近似 |

统一注意事项：

1. 复权：所有跨日比率/滚动窗口特征必须用前复权（或后复权一致口径）价格计算，避免除权日伪造跳变；qlib 社区惯例是用复权因子调整后的价格。
2. 截面 rank 类特征（Alpha101/191 的 RANK、BigQuant rank_*）在"信号池内排序"场景可直接在当日信号池内计算，不引入未来信息。
3. 异常成交量类因子（GTJA 研报原文提示）需剔除涨跌停日造成的量异常，否则信号失真。
4. vwap 字段：无现成 vwap 时用 `amount/volume`（A 股日线两个字段都有）；停牌日 amount/volume 为 0 需做缺失处理。
5. 窗口预热：最长窗口 60 日（Alpha158）/ 递推类（EMA/MACD 需约 3 倍窗口预热，Alpha191 的 SMA/WMA 递推需 60+ 日），特征矩阵前 ~90 行不可用于训练。

## 13. 本场景 Top 20 推荐（底背离事件池内排序，5-30 日持仓）

选型逻辑：信号本身已是"价格新低 + 动量不新低"的多头反转赌注，二次排序需要的是 (a) 超跌深度与位置、(b) 底部确认（缩量/价量背离/波动收敛）、(c) 反弹弹性与流动性、(d) 与事件窗口匹配的周期（5/10/20/30 日，弃用 60+）。

| # | 特征 | 公式（T 日口径） | 来源 | 排序含义 |
|---|---|---|---|---|
| 1 | RSV_9 | `(close-LLV(low,9))/(HHV(high,9)-LLV(low,9))` | qlib RSV / KDJ | 价格在 9 日区间的位置，越低越超跌 |
| 2 | BIAS_10 | `(close-MA(close,10))/MA(close,10)` | 聚宽/米筐/Alpha31 | 负乖离深度，短期反转核心 |
| 3 | ROC_20（反转口径） | `close/Ref(close,20)-1`（取负值大者） | qlib ROC / Alpha191 | 20 日超跌幅度 |
| 4 | 量幅背离 | `-CORR(high/low, volume, 10)` | GTJA 研报（ICIR 4.26，全研报最强） | 振幅缩+量缩的底部确认 |
| 5 | 价量背离 | `-CORR(vwap, volume, 10)` | GTJA 研报 / Alpha191 | 价跌量增衰竭确认 |
| 6 | 异常成交量（缩量） | `volume/MEAN(volume,20)`（剔除涨跌停日） | GTJA 研报 | <1 为缩量，底部特征 |
| 7 | VMA_5（量比） | `MEAN(volume,5)/volume` 或 `volume/MA(volume,5)` | qlib VMA / 通达信量比 | 当日量能相对水平 |
| 8 | RANK_20（时序分位） | `ts_rank(close,20)` | qlib RANK / RQFactor TS_RANK | 收盘价在 20 日内分位，低=贴底 |
| 9 | IMIN_20 | `LLVBARS(low,20)/20`（=IdxMin/20） | qlib IMIN / MyTT | 最低点距今天数，刻画背离结构新鲜度 |
| 10 | IMXD_20 | `(IdxMax(high,20)-IdxMin(low,20))/20` | qlib IMXD | 先高后低（负值）= 下跌结构完整 |
| 11 | SUMD_20 | `SUMP20-SUMN20`（RSI 居中变体） | qlib SUMD | 涨跌力量不对称，深负=超卖 |
| 12 | RSI_6 / SUMP_10 | `RSI(close,6)` | TA-Lib / 米筐 RSI6 | 经典超卖度量 |
| 13 | CNTD_20 | `COUNT(涨,20)/20 - COUNT(跌,20)/20` | qlib CNTD / Alpha53 | 连跌天数结构 |
| 14 | MACD_HIST | `MACD(close,12,26,9)` 柱值及 Δ柱 | TA-Lib / 全平台 | 与背离事件同源，柱值回升确认 |
| 15 | NATR_14 | `ATR(14)/close` | TA-Lib | 归一化波动，用于仓位/止损幅度与弹性估计 |
| 16 | STD_20（波动收敛） | `Std(close,20)/close` | qlib STD / 米筐 VOLT20 | 波动收敛的底部特征，也用于过滤 |
| 17 | 开盘缺口 | `open/Ref(close,1)-1` | GTJA 研报（IR 4.83） | T+1 买入日的跳空风险/动量参考（T 日值可算） |
| 18 | BOLL_%b | `(close-BOLL_DOWN(20,2))/(BOLL_UP-BOLL_DOWN)` | TA-Lib BBANDS | 贴下轨程度 |
| 19 | 换手率能级 | `MEAN(turnover,5)/MEAN(turnover,60)`（DAVOL 变体） | 聚宽 DAVOL / 米筐 VOL | 资金关注度退潮/维持，反弹持续性代理 |
| 20 | Alpha101#101 实体强度 | `(close-open)/(high-low+1e-12)`（=qlib KMID2） | WorldQuant Alpha101 | T 日 K 线质量（下影+收阳优先，可与 KLOW2 组合） |

备选（第 21-30 顺位）：`STOCHRSI`、`MFI_14`（量价超卖）、`CYR`（米筐市场强弱）、`WVMA_20`（qlib 量加权波动）、`WR_10`、`PPO`（跨股可比 MACD）、`DECAYLINEAR(returns,10)` 加权短期收益、`CORR(close,log(volume),20)`（qlib CORR20）、锤头/启明等反转型 CDL 形态、`BARSLAST(前低)` 结构时间。

## 14. 程序化生成建议

1. 生成器 = 基础字段 × 单元变换 × (窗口 × rolling 算子) × 后处理的笛卡尔积，按第 2.4/8.1/9.1 节的算子表实现一次即可覆盖 Alpha158 + Alpha101/191 的绝大部分。
   - 基础字段：`open, high, low, close, volume, amount, vwap(=amount/volume), turnover, returns(=close/ref(close,1)-1), high/low, log(volume+1)`。
   - rolling 算子：`Ref, Mean, Sum, Std, Var, Skew, Kurt, Max, Min, Med, Quantile(q), Rank(ts), IdxMax, IdxMin, Slope, Rsquare, Resi, Delta, EMA, WMA/DecayLinear, Count(cond), Corr(双序列), Cov`。
   - 窗口：`{3, 5, 10, 20, 30, 60}`（5-30 日持仓主用 3/5/10/20/30）。
   - 后处理：`/close、/open、-1×、rank(截面)、ts_zscore`。
2. 直接用 KunQuant 或 qweave 可免写引擎：二者均内置 Alpha101/158/191 的批量表达式生成。
3. 命名与审计：沿用 qlib 命名（`前缀+窗口`），便于与社区结果对表；每个特征记录表达式字符串进特征注册表，防止口径漂移。
4. 与本仓库 V4 教训的衔接：严禁把任何含 T 之后信息的量（如 stop_loss_return_* 类未来收益）混入特征表；上表全部特征只依赖 ≤T 数据，落地时仍需对生成器做逐列 leakage 审计（shift 对齐单测）。
