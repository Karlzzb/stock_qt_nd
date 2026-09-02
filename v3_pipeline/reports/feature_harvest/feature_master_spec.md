# 特征主表（Feature Master Spec）v1.0

> 本文件是后续特征引擎的**唯一实现依据**。
> 由五份收割报告合并而成：academic_factors.md（A）、community_factors.md（C）、event_context_factors.md（E）、legacy_features.md（L）、local_inventory.md（数据契约）。
> 任何特征的增删改必须先改本表再改代码；代码与本表不一致时以本表为准。
> 生成日期：2026-09-01。

---

## 0. 术语说明（新词一律在此定义）

- **特征工厂层**：不手写单个特征，而是定义"字段 × 变换 × 算子 × 窗口"四类积木，由程序批量生成全部合法组合（如 收盘价 × 与成交量的滚动相关 × 20 日 = 近 20 日量价相关）。
工厂负责"一个不漏"，稳定性三关筛负责"只留真金"；机器生成的特征一律不带先验。
- **P0 / P1 / P2**：精编特征的优先级层。P0 = 证据强且本地可算（首批手写实现）；P1 = 可算但证据中或弱（第二批）；P2 = 需补外部数据。
- **截断对拍**：泄漏结构测试。把历史数据在信号日 T 处截断后重算特征，与全历史计算结果比对，必须零差异，证明特征只用了 T 及之前的数据。
- **稳定性三关筛**：特征准入门槛——泄漏结构关（截断对拍）、分年稳定关（逐年表现不剧烈漂移）、共线性关（与已有特征高度冗余的剔除）。
- **横截面变体**：同一特征在全市场当日截面上的排名分位（`_csrank`）或标准化（`_csz`）变换版本。
- **CF 复权链**：用每日涨跌幅累乘重构的复权价格序列（见 1.2），解决本地数据为不复权价导致的跨日价格比较失真。

## 1. 数据契约与记号

### 1.1 本地字段（唯一允许的数据源）

个股日线 `stock_data/daily/*.parquet`（tushare daily 原样，**不复权**）：
`ts_code, trade_date, open(O), high(H), low(L), close(C), pre_close(PC), change, pct_chg, vol(V,手), amount(A,千元)`。
指数日线：`000001.SH`（上证综指）、`399001.SZ`（深证成指），仅 OHLC+volume，无 amount/pre_close。
universe：`universe_latest.parquet`（ts_code / name / list_date / delist_date）。
除此之外**无任何其他数据**（无换手率、市值、行业、资金流、分钟线、涨跌停价表、历史 ST 状态）。

### 1.2 除权失真处理决定（全引擎统一口径）

**决定：用 `pct_chg` 链重构复权序列，所有跨日价格比较一律在复权链上计算；不补外部复权因子，不弃用价格水平类特征。**

理由一句话：`pct_chg`/`pre_close` 已含除权修正，链式累乘是零外部依赖、且已在 divergence_lab 的 `cf` 因子中全量验证过的复权口径。

具体口径：

- `R_t = pct_chg_t / 100`：真实日收益（含除权修正）。
- `CF_t = cumprod(1 + R_t)`（逐股、按 trade_date 排序、自该股首个可得日起锚定 1）：复权收益链。
- 日度调整因子 `f_t = CF_t / C_t`；虚拟复权 OHLC：`C̃=CF, Õ=O×f, H̃=H×f, L̃=L×f`。
- **用复权价的特征**：一切跨日价格比——收益/动量（RET*）、均线与乖离（BIAS*、MA_STACK、TREND_SLOPE）、距高/低点（DIST_*、POS_52W、HIGH52 类）、价格分位/时序秩、MACD/RSI/KDJ/BOLL 等全部技术指标、量价相关中的价格腿。
- **用原始价的特征**：单日截面内的比率——K 线形态（实体/影线/振幅）、`RET_ID=C/O-1`、AMP1；以及跳空 `RET_ON=O/PC-1`（pre_close 本身就是除权修正锚，天然安全）。
- **保留原始价并标注失真的特征**：`LOG_PRICE=ln(C)` 仅此一条，仅作彩票偏好/面值退市代理；除权日会产生不可比跳变，建模时不得与复权价特征混用解读。
- vol/amount 无需调整，直接用。

### 1.3 公式记号

- 滚动算子均为**逐股 groupby(ts_code)**：`Mean(x,w)`、`Std(x,w)`、`Sum(x,w)`、`Max(x,w)`、`Min(x,w)`、`Med(x,w)`、`Quant(x,w,q)`、`Skew(x,w)`、`Kurt(x,w)`、`TsRank(x,w)`（最新值在 w 日内分位）、`IdxMax/IdxMin(x,w)`（最值距今天数）、`Slope(x,w)`（对时间 OLS 斜率）、`Corr(x,y,w)`、`EMA(x,w)`、`Count(cond,w)`、`shift(x,n)`。
- `zscore_ts(x,w) = (x - Mean(x,w)) / Std(x,w)`；`pct_ts(x,w)` = 当前值在过去 w 日的时序分位。
- 横截面算子（按 trade_date 分组，仅当日快照）：`cs_rank(x)`（百分位）、`cs_z(x)` = (x−median)/std。
- 指数序列加 `IDX` 前缀，默认 000001.SH；指数收益由 close 自行差分（指数文件无 pct_chg）。
- 所有特征**信号日 T 收盘后可算**，即只用 t ≤ T 的数据；最长窗口 250 交易日（52 周类），递推类（EMA/MACD/KDJ）预留 3 倍窗口预热。
- 事件特征中的背离结构记号沿用 E 报告：第一/第二价格低点 `i1/i2`，确认日 T，`DIF/DEA/HIST` 为 MACD(12,26,9) 国内口径，`ATR14` 为 Wilder ATR，`ATRp = ATR14/C̃`。

### 1.4 来源与证据强度标记

- 来源：`A`=学术、`C`=社区、`E`=事件语境、`L`=历史（v2 管线/divergence_lab/architecture_race）。
- 证据强度：**强** = A 股学术实证有显著统计量，或历史 walk-forward 稳定名单/V4 干净模型验证，或多源独立收敛；**中** = 单源实证或机制明确但量级不稳；**弱** = 方向性假设/证据不一致。
- 优先级：**P0** = 证据强且可算（引擎首批手写实现）；**P1** = 可算但证据中/弱（第二批）；**P2** = 需补数据；**G** = 特征工厂层（第 5 章）。

---

## 2.【可算】特征主表 —— P0（60 条）

P0 是引擎首批手写实现对象，全部落到本地字段、T 收盘可算。
标注 ⟂ 的为冗余对成员（建模时二选一或正交化，见 2.3）。

### 2.1 P0 表

| # | 规范名 | 公式（本地字段口径） | 家族 | 来源 | 证据 |
|---|---|---|---|---|---|
| 1 | `RET1` | `R`（= pct_chg/100） | 反转 | A,C,L | 强 |
| 2 | `RET5` | `CF/shift(CF,5)-1` | 反转 | A,C,L | 强 |
| 3 | `RET10` | `CF/shift(CF,10)-1` | 反转 | C,L | 强 |
| 4 | `RET20` ⟂ | `CF/shift(CF,20)-1` | 反转 | A,C,E,L | 强 |
| 5 | `RET_ON` | `O/PC - 1` | 隔夜/日内 | A,L | 强 |
| 6 | `RET_ID` | `C/O - 1` | 隔夜/日内 | A,L | 强 |
| 7 | `CUMON20` | `Sum(log1p(RET_ON), 20)` | 隔夜/日内 | A | 强 |
| 8 | `CUMID20` | `Sum(log1p(RET_ID), 20)` | 隔夜/日内 | A | 强 |
| 9 | `VOL5` | `Std(R,5)` | 波动率 | L | 强 |
| 10 | `VOL15` | `Std(R,15)` | 波动率 | L | 强 |
| 11 | `VOL20` ⟂ | `Std(R,20)` | 波动率 | A | 强 |
| 12 | `IVOL60` | 60 日 `R = α+β·R_idx` OLS 残差的 std（指数 000001.SH） | 波动率 | A | 强 |
| 13 | `AMP1` | `(H-L)/PC` | 波动率 | C,L | 强 |
| 14 | `AMP20` ⟂ | `Mean((H-L)/PC, 20)` | 波动率 | A | 强 |
| 15 | `ATRN` | `ATR14(H,L,C)/C̃`（Wilder 14） | 波动率 | A,L | 强 |
| 16 | `ILLIQ20` | `log(Mean(abs(R)/(A×1000), 20) + 1e-30)`（amount 千元→元） | 流动性 | A,L | 强 |
| 17 | `AMT20` | `log(Mean(A,20))` | 流动性 | A,L | 强 |
| 18 | `VR5_60` | `Mean(V,5)/Mean(V,60)` | 量能 | A,C,E | 强 |
| 19 | `VR1_20` | `V/Mean(V,20)` | 量能 | C,L | 强 |
| 20 | `VOL_MOM` | `(Mean(V,5)-Mean(V,20))/Mean(V,20)` | 量能 | L | 强 |
| 21 | `CPV10` | `Corr(C̃, V, 10)` | 量价相关 | C,L | 强 |
| 22 | `CPV20` | `Corr(C̃, V, 20)` | 量价相关 | A,C | 强 |
| 23 | `CPV_VWAP10` | `-Corr(A/V, V, 10)`（vwap=amount/vol） | 量价相关 | C | 强 |
| 24 | `HLV_DIV10` | `-Corr(H/L, V, 10)` | 量价相关 | C | 强 |
| 25 | `MAX20` ⟂ | `Max(R,20)` | 彩票 | A | 强 |
| 26 | `MIN20` | `Min(R,20)` | 彩票 | A | 强 |
| 27 | `MKT_RET20` | `C_idx/shift(C_idx,20)-1` | 市场状态 | A,E | 强 |
| 28 | `MKT_MA60` | `C_idx/Mean(C_idx,60)-1` | 市场状态 | A,E | 强 |
| 29 | `MKT_AVG_AMP` | 两指数 `(H-L)/L` 当日均值 | 市场状态 | L | 强 |
| 30 | `MKT_SYNC_SCORE` | 两指数日收益同号强度（L 口径：`market_sync_score`） | 市场状态 | L | 强 |
| 31 | `LIMITCNT20` ⟂ | `Count(pct_chg ≥ limit-0.1, 20)`，limit 按板块取 10/20/30 | 制度 | A | 强 |
| 32 | `LIMITDOWN_CNT20` | `Count(pct_chg ≤ -(limit-0.1), 20)` | 制度 | A | 强 |
| 33 | `MACD_DIF_NORM` | `DIF/(C̃×ATRp)` | 技术指标 | A,E,L | 强 |
| 34 | `MACD_HIST` | `2×(DIF-DEA)` | 技术指标 | C,L | 强 |
| 35 | `MACD_HIST_SLOPE5` | `Slope(MACD_HIST, 5)` | 技术指标 | C,L | 强 |
| 36 | `MACD_ZERO_CROSS` | DIF 穿零轴标记（上穿+1/下穿-1，仅穿越日为非零） | 技术指标 | L | 强 |
| 37 | `RSI6` | Wilder RSI(C̃,6) | 技术指标 | A,C,L | 强 |
| 38 | `RSI14` | Wilder RSI(C̃,14) | 技术指标 | A,C,E | 强 |
| 39 | `RSI_DIV` | `1[L̃[i2]<L̃[i1] 且 RSI14[i2]>RSI14[i1]]`（事件行） | 事件结构 | A,E | 强 |
| 40 | `BIAS20` ⟂ | `C̃/Mean(C̃,20)-1` | 技术指标 | A,C,E | 强 |
| 41 | `BIAS60` | `C̃/Mean(C̃,60)-1` | 技术指标 | E,L | 强 |
| 42 | `BOLL_SQUEEZE` | `1[(4×Std(C̃,20))/Mean(C̃,20) < 0.05]` | 技术指标 | L | 强 |
| 43 | `DIST_SUPPORT20` | `(C̃-Min(L̃,20))/C̃` | 位置结构 | L | 强 |
| 44 | `DIST_RESIST20` | `(Max(H̃,20)-C̃)/C̃` | 位置结构 | L | 强 |
| 45 | `SR_RATIO20` | `Mean(H̃,20)/Mean(L̃,20)`（RSRS 思路） | 位置结构 | L | 强 |
| 46 | `EFF_RATIO10` | `abs(C̃-shift(C̃,10))/Sum(abs(ΔC̃),10)`（Kaufman） | 位置结构 | L | 强 |
| 47 | `DIST_HIGH60` | `C̃/Max(C̃,60)` | 位置结构 | E,L | 强 |
| 48 | `CLOSE_VS_HIGH` | `(H-C)/(H-L)`，一字板置 0.5 | K线形态 | L | 强 |
| 49 | `PRICE_TREND5` | `Slope(C̃,5)/C̃` | 位置结构 | C,L | 强 |
| 50 | `MA_STACK` | `sign(C̃-MA60)+sign(MA60-MA120)+sign(MA120-MA250)`（三分量或加总，实现时定） | 趋势 | E,L | 强 |
| 51 | `TREND_SLOPE_120` | `Slope(log(C̃),120)×120 / (Std(R,120)×sqrt(120))`（t 统计量口径） | 趋势 | E,L | 强 |
| 52 | `DIV_COUNT_120` | 过去 120 日内同向底背离次数（含本次） | 事件结构 | E | 强 |
| 53 | `DIV_HIST_AREA_SHRINK` | `S2/S1`，Sk = 第 k 低点前绿柱区间 `Sum(HIST|HIST<0)` | 事件结构 | E | 强 |
| 54 | `DIV_GOLDEN_CROSS_STATE` | 三值：2=T 日金叉；1=T 前 5 日内已金叉；0=未金叉 | 事件结构 | E | 强 |
| 55 | `REBOUND_FROM_L2` | `(C̃[T]-L̃[i2])/ATR14[T]` | 事件结构 | E | 强 |
| 56 | `DIV_PRICE_NEWLOW_DEPTH` | `L̃[i2]/L̃[i1]-1` | 事件结构 | E,L | 强 |
| 57 | `DIV_DIF_LIFT` | `(DIF[i2]-DIF[i1])/ATR14[T]` | 事件结构 | E,L | 强 |
| 58 | `DIV_SPAN_BARS` | `i2-i1`（= L 的 formation_period） | 事件结构 | E,L | 强 |
| 59 | `DAYS_SINCE_L2` | `T-i2`（= L 的 confirm_lag） | 事件结构 | E,L | 强 |
| 60 | `RET20_CSR` | `cs_rank(RET20)`（横截面跌幅分位，E 的 B9 由本条承接） | 反转×截面 | A,E,L | 强 |

### 2.2 P0 备注

- `ATRN` 是标签阈值单位（+2×ATR 狙击标签），属于"标签几何"控制变量，必须入模但解读时与 alpha 区分；`VOL20/AMP20` 与标签可达性存在机械相关，同理处理（A 报告实施备注 2）。
- `RET20_CSR` 代表"横截面后处理承接位"：所有 P0/P1 数值特征自动派生 `_csrank`/`_csz` 变体（见 4.3），本行单独成条是因为 E(B9) 明确提出跌幅的横截面/时序双分位；时序分位版 `pct_ts(RET20,250)` 归特征工厂层。
- `LIMITCNT20/LIMITDOWN_CNT20` 的板块涨停幅度按 ts_code 前缀判定：688→20%、300→20%（2020-08-24 起，此前 10%）、8xx/4xx/920→30%、其余 10%；ST 的 5% 无法识别（无历史 ST 状态），误差接受并在特征注册表注明。无 stk_limit 价格表，属近似口径，升级路径见 P2-14。
- 事件结构类（39、52-59）只在事件行（events.parquet 对齐行）有定义，非事件行为 NaN；低点锚定只用 v2 检测器 left3/right2 或 divergence_lab fractal/zigzag（见 6.3）。

### 2.3 冗余对处置（A 报告标的 6 对 + 合并中发现）

| 冗余对 | 处置 |
|---|---|
| `RET20` ↔ `BIAS20`（A 1.1↔13.4） | 均入 P0，公式基准不同（点对点 vs 均线），筛选阶段按共线性规则保留其一 |
| `AMP20` ↔ `VOL20`（A 2.4↔2.1） | 均入 P0，筛选时正交化或二选一 |
| `TURN20` ↔ `ILLIQ20` ↔ `FLOAT_MCAP`（A 3.1↔3.4↔3.7） | 可算的 `ILLIQ20` 入 P0 作流动性+规模代理；另两条入 P2，补数据后按共线性裁决 |
| `MAX20` ↔ `LIMITCNT20`（A 7.1↔12.1） | 均入 P0，互补（幅度 vs 制度封板），不合并 |
| `REL_REV` ↔ `REL_STR`（A 1.6↔10.3） | 合并为一条 `IND_REL_STR`，入 P2 |
| `DOW` ↔ `TRADEDAY_SEQ`（A 8.1↔8.5） | 合并为 `CAL_DOW`（one-hot），入 P1 |
| `RET20` ↔ 乖离率族、`MAX` ↔ 涨停次数（社区重叠） | 已在主表合并，社区同义式（qlib ROC/MA、Alpha31/66/71、Alpha53 等）不再单列 |
| `CLV/intraday_pos` ↔ `CLOSE_VS_HIGH` | `(C-L)/(H-L) = 1 - (H-C)/(H-L)`，完全线性冗余，只保留 `CLOSE_VS_HIGH` |
| `NATR` ↔ `ATRN` | 同式，只保留 `ATRN` |
| `vol_dryup_5_60`(E B12) ↔ `VR5_60` | 同式，并入 #18 |
| `drawdown_60`(E B10) ↔ `DIST_HIGH60` | 同义（C̃/Max(H̃,60)-1 ≈ C̃/Max(C̃,60)），并入 #47，实现取 high 口径 |
| `gap_down_T`(E B11) ↔ `RET_ON` | 并入 #5；过去 20 日最大跳空归特征工厂层 |
| `vol_price_div`(E E9) ↔ `VSHRINK` | 并入 P1 `VSHRINK` |
| `atrp_level`(E B6) ↔ `ATRN` | 水平并入 #15；`pct_ts(ATRp,250)` 归特征工厂层 |

---

## 3.【可算】特征主表 —— P1（80 条）

证据中/弱但可算，引擎第二批实现；其中生成器可覆盖的（标注 G-ok）不必手写，直接从特征工厂层产出。

| # | 规范名 | 公式（本地字段口径） | 家族 | 来源 | 证据 | 备注 |
|---|---|---|---|---|---|---|
| 1 | `RET60` | `CF/shift(CF,60)-1` | 反转 | C | 中 | G-ok |
| 2 | `RET120_20` | `shift(CF,20)/shift(CF,240)-1`（剔除近 1 月的中期动量） | 动量 | A | 中 | 方向不稳，作控制变量 |
| 3 | `GAP_MEAN20` | `Mean(RET_ON,20)` | 隔夜 | A | 中 | 与 CUMON20 近义 |
| 4 | `GAP_VOL20` | `Std(RET_ON,20)` | 隔夜 | A | 中 | |
| 5 | `INFO_DISC20` | `sign(RET20)×(Count(R<0,20)-Count(R>0,20))/20` | 反转 | A | 中 | Frog-in-the-Pan |
| 6 | `T1_GAP` | `Mean(RET_ON[t] | RET_ID[t-1] < -2%, 20)` | 制度 | A | 弱 | 交互项 |
| 7 | `VOL60` | `Std(R,60)` | 波动率 | A,C | 中 | G-ok |
| 8 | `DVOL` | `Std(R,5)-Std(R,60)` | 波动率 | A,L | 中 | |
| 9 | `PARK20` | `sqrt(Sum(log(H/L)^2,20)/(4×ln2×20))` | 波动率 | A | 中 | 涨跌停日失真 |
| 10 | `GK_VOL` | `sqrt(0.5×ln(H/L)^2-(2ln2-1)×ln(C/O)^2)` 的 20 日均值 | 波动率 | L | 中 | Garman-Klass |
| 11 | `GK_VOL_RATIO` | `GK_VOL/Mean(GK_VOL,20)` | 波动率 | L | 中 | |
| 12 | `VOL_CONTRACTION60` | `ATRp/shift(ATRp,60)` | 波动率 | E | 中 | |
| 13 | `BBW_PCTILE250` | `pct_ts(4×Std(C̃,20)/Mean(C̃,20), 250)` | 波动率 | E | 中 | 挤压分位 |
| 14 | `VMA5` | `V/Mean(V,5)`（量比近似） | 量能 | C | 中 | G-ok |
| 15 | `VSTD20` | `Std(V,20)/(Mean(V,20)+1)` | 量能 | C,L | 中 | G-ok |
| 16 | `VOLUME_TREND10` | `Slope(V,10)/Mean(V,20)` | 量能 | L | 中 | |
| 17 | `OBV_TREND` | `Slope(OBV,5)/Mean(V,20)`，OBV 按 sign(R)×V 累积 | 量能 | L | 中 | 累积量必须归一 |
| 18 | `OBV_DIV` | `zscore_ts(C̃,20)-zscore_ts(OBV,20)` | 量价背离 | A | 中 | |
| 19 | `PVR20` | `Mean(V|R>0,20)/Mean(V|R<0,20)`（最少 5 个上涨日否则 NaN） | 量价背离 | A | 中 | |
| 20 | `VSHRINK` | `Mean(V|R<0,近10)/Mean(V|R<0,前10)` | 量价背离 | A,E | 中 | 吸收 E9 |
| 21 | `VOL_DRYUP_EXTREME` | `1[Mean(V,5) < Quant(Mean(V,5),250,0.1)]` | 量能 | E | 中 | |
| 22 | `AMT_SHRINK_PEAK` | `Mean(A,5)/Max(Mean(A,5),120)` | 量能 | E | 中 | |
| 23 | `RVC20` | `Corr(R, V/shift(V,1)-1, 20)` | 量价相关 | A,C | 中 | |
| 24 | `CPV_TREND` | `Slope(Corr(C̃,V,10), 10)` | 量价相关 | A | 弱 | 噪声大 |
| 25 | `DSR20` | `Sum(min(R,0)^2,20)/Sum(R^2,20)` | 波动结构 | A | 中 | 日频近似 |
| 26 | `SJV60` | `Sum(R^2·1[R>θ],60)-Sum(R^2·1[R<-θ],60)`，`θ=2×Std(R,60)` | 波动结构 | A | 弱 | |
| 27 | `JUMPFREQ60` | `Count(abs(R)>2×Std(R,60),60)/60` | 波动结构 | A | 中 | |
| 28 | `MAX5_20` | 20 日内最大 5 个 R 的均值 | 彩票 | A | 中 | |
| 29 | `SKEW60` | `Skew(R,60)` | 高阶矩 | A | 中 | |
| 30 | `KURT60` | `Kurt(R,60)` | 高阶矩 | A | 弱 | 仅作交互 |
| 31 | `IVOV60` | `Std(Std(R,5),60)` | 高阶矩 | A | 中 | |
| 32 | `CAL_DOW` | 信号日星期 one-hot（5 列） | 日历 | A | 中 | 吸收 TRADEDAY_SEQ |
| 33 | `CAL_MONTH_POS` | 距月末交易日数（连续值） | 日历 | A | 弱 | |
| 34 | `CAL_HOLIDAY` | 春节/国庆前 10 日、后 5 日 dummy | 日历 | A | 弱 | 样本少 |
| 35 | `CAL_MONTH` | 月份 one-hot | 日历 | A | 弱 | 作交互 |
| 36 | `MKT_VOL20_PCT` | `pct_ts(Std(R_idx,20),250)` | 市场状态 | A,E | 中 | |
| 37 | `MKT_DD120` | `C_idx/Max(H_idx,120)-1` | 市场状态 | E | 中 | |
| 38 | `MKT_RSI14` | `RSI(C_idx,14)` | 市场状态 | E | 中 | |
| 39 | `BREADTH_ADV5` | universe 内 `RET5>0` 占比 | 市场宽度 | E | 中 | 自算，无外部依赖 |
| 40 | `BREADTH_NEWLOW` | universe 内 `L = Min(L,250)` 占比 | 市场宽度 | E | 中 | |
| 41 | `BREADTH_ABOVE_MA20` | universe 内 `C̃>Mean(C̃,20)` 占比 | 市场宽度 | E | 中 | |
| 42 | `MKT_MEDIAN_RET20` | universe 内 RET20 中位数 | 市场宽度 | E | 中 | |
| 43 | `LIMITUP_N_MKT` | 全市场涨停家数（pct_chg≥limit-0.1）及其 20 日均值 | 市场情绪 | A | 中 | 同 #31 近似口径 |
| 44 | `MKT_AMT_PCT` | 全市场 amount 加总 20 日均值的 250 日分位 | 市场情绪 | A | 中 | RETAIL_SENT 的无股本代理 |
| 45 | `MKT_IDX_DAILY` | 双指数日度族：`(C-O)/O`、`(H-L)/L`、量比、绝对值、情绪哑变量、同步度（L 2.13 节 20 列口径） | 市场状态 | L | 中 | 整族复用 v2 公式重写 |
| 46 | `REGIME_CODE` | 全样本等权日收益累积指数 120 日滚动收益 → unknown/sideways/up/down → -1/0/1/2 | 市场状态 | L | 中 | 复用 divergence_lab `build_market_regime` |
| 47 | `DIST_LIMIT` | `pct_chg/limit_pct`（∈[-1,1]，1=收涨停） | 制度 | A | 中 | 磁吸效应 |
| 48 | `DISPOSAL60` | `C / (Sum(A,60)/Sum(V,60)) - 1`（现价对 60 日 VWAP 持仓成本的偏离；amount/vol 的单位常数差不影响比率形态，实现时统一换算） | 制度 | A | 中 | 解套抛压结构 |
| 49 | `NEW_LISTING` | `1[T - list_date < 250 交易日]` | 制度 | A | 中 | list_date 来自 universe_latest |
| 50 | `PCTB` | `(C̃-Mean(C̃,20))/(2×Std(C̃,20))`（%B） | 技术指标 | A,C | 中 | |
| 51 | `KDJ_J` | RSV9 递推 K/D 后 `J=3K-2D`（输出 K/D/J 三列） | 技术指标 | C,E,L | 中 | |
| 52 | `RSI_OVERSOLD_DAYS` | RSI14<30 连续天数 | 技术指标 | A | 中 | |
| 53 | `RSV9` | `(C̃-Min(L̃,9))/(Max(H̃,9)-Min(L̃,9))` | 技术指标 | C | 中 | G-ok |
| 54 | `CNTD20` | `Count(R>0,20)/20 - Count(R<0,20)/20` | 技术指标 | C | 中 | G-ok |
| 55 | `SUMD20` | `Sum(max(R,0),20)/Sum(abs(R),20) - Sum(max(-R,0),20)/Sum(abs(R),20)` | 技术指标 | C | 中 | G-ok |
| 56 | `TSRANK20` | `TsRank(C̃,20)` | 位置 | C | 中 | G-ok |
| 57 | `IMIN20` | `IdxMin(L̃,20)/20` | 位置 | C | 中 | G-ok |
| 58 | `IMXD20` | `(IdxMax(H̃,20)-IdxMin(L̃,20))/20` | 位置 | C | 中 | G-ok |
| 59 | `MACD_HIST_AMP20` | `Max(HIST,20)-Min(HIST,20)` | 技术指标 | L | 中 | |
| 60 | `MACD_GOLDEN_CROSS` | DIF 上穿 DEA 标记（日频，非事件窗） | 技术指标 | L | 中 | |
| 61 | `HAMMER` | 下影≥2×实体 且 上影≤0.5×实体 且 下影≥0.6×全长（T 日与 i2 日各一列） | K线形态 | E,L | 中 | |
| 62 | `LOWER_SHADOW_L2` | `(min(O,C)-L)[i2]/ATR14[i2]` | K线形态 | E | 中 | |
| 63 | `DOWN_STREAK` | T 前最长连续 C<O 天数；及 10 日阴线占比 | K线形态 | E | 中 | 非单调 |
| 64 | `UPPER_SHADOW_RATIO` | `(H-max(O,C))/(H-L)`，一字板置 0 | K线形态 | L | 中 | |
| 65 | `BODY_STRENGTH` | `(C-O)/(H-L+1e-12)` | K线形态 | C,L | 中 | G-ok |
| 66 | `DOJI` | `abs(C-O) < 0.1×(H-L)` | K线形态 | L | 弱 | |
| 67 | `ENGULFING` | 前阴今阳/前阳今阴且实体吞没 | K线形态 | C,L | 中 | |
| 68 | `POS_52W` | `(C̃-Min(L̃,250))/(Max(H̃,250)-Min(L̃,250))` | 位置 | A,E | 中 | 非单调 |
| 69 | `DIST_52W_LOW` | `C̃/Min(L̃,250)-1` | 位置 | E | 中 | |
| 70 | `DIST_52W_HIGH` | `C̃/Max(H̃,250)-1` | 位置 | A,E | 弱 | A 股 52 周高点效应证据负面，仅作位置控制 |
| 71 | `MA200_RATIO` | `C̃/Mean(C̃,200)-1` 及 `1[C̃>MA200]` | 趋势 | L | 中 | |
| 72 | `DIV_SPAN_VS_CYCLE` | `(i2-i1)/最近一次 DIF 完整正负循环时长` | 事件结构 | E | 弱 | |
| 73 | `DIV_DIF_SLOPE` | `(DIF[i2]-DIF[i1])/(i2-i1)/ATR14[T]` | 事件结构 | E | 中 | |
| 74 | `DIV_ZERO_AXIS_DEPTH` | `max(DIF[T],DEA[T])/(C̃[T]×ATRp[T])` | 事件结构 | E | 中 | 非单调 |
| 75 | `DIV_DIF_LEVEL_L1` | `DIF[i1]/(C̃[i1]×ATRp[i1])` | 事件结构 | E,L | 中 | |
| 76 | `DIV_HIST_TROUGH_SHRINK` | `min(HIST@i2 绿柱区)/min(HIST@i1 绿柱区)` | 事件结构 | E | 中 | |
| 77 | `REBOUND_DAY_T` | `(C[T]-O[T])/ATR14[T]` | 事件结构 | E | 中 | |
| 78 | `AMIHUD_INTRADAY` | `abs(C/O-1)/(V×C)` | 流动性 | L | 中 | 日内口径 |
| 79 | `HL_SPREAD` | `(H-L)/((H+L)/2)` | 流动性 | L | 中 | G-ok |
| 80 | `PRICE_IMPACT` | `abs(C-O)/sqrt(V)` | 流动性 | C,L | 中 | WQ Alpha 族 |
| 81 | `LOG_PRICE` | `ln(C)`（**原始价，除权失真标注**） | 价格水平 | A | 弱 | 仅作彩票/退市代理 |

> `ALPHA12`（sign(ΔV)×(−ΔC)）与 `VWAP_DEV`（C/(A/V)−1）等 WQ/qlib 单式变体不再单列，由特征工厂层覆盖。
> K 线形态族（TA-Lib CDL\* 61 个）不单列，反转型形态（锤头/倒锤/启明/看涨吞没/蜻蜓十字）若需要，由特征工厂层以 `1[形态]` 算子批量产出，全部按机器生成特征对待。

---

## 4.【需补数据】清单（P2，15 条）

补数据后按本表公式实现；未补之前不得用近似口径冒充（涨跌停、市场情绪除外——已有显式标注的近似版在 P0/P1）。

| # | 规范名 | 公式要点 | 缺什么 | 来源 |
|---|---|---|---|---|
| 1 | `TURN20` | `Mean(V/流通股本, 20)` | 流通股本（tushare daily_basic 或 akshare） | A |
| 2 | `ABTURN` | `Mean(turn,5)/Mean(turn,60)` | 同上 | A |
| 3 | `STDTURN20` | `Std(turn,20)` | 同上 | A |
| 4 | `FLOAT_MCAP` | `ln(C×流通股本)` | 同上 | A |
| 5 | `RETAIL_SENT` | 全市场换手率加总 20 日均值的 250 日分位（去趋势） | 同上 | A |
| 6 | `IDEAL_REV` | 20 日按单笔金额（A/成交笔数）分组切割的反转差 | 成交笔数 | A |
| 7 | `IND_REL_STR` | `RET20_stock - RET20_industry`（合并 A 1.6 与 10.3） | 行业分类（申万/中信，需 T 时点历史成分） | A,E |
| 8 | `IND_MOM20` | 行业等权指数 20 日收益 | 同上 | A,E |
| 9 | `IND_RS_RANK` | 行业 20 日收益的横截面分位 | 同上 | A |
| 10 | `IND_BREADTH_MA20` | 行业内 C̃>MA20 成分占比 | 同上 | E |
| 11 | `IND_RET5_RANK` | 行业 5 日收益的横截面分位 | 同上 | E |
| 12 | `ST_FLAG` | ST/*ST/退市整理 dummy（T 时点历史状态） | 历史 ST 状态表（universe 仅当前名称，不可用） | A |
| 13 | `MKT_CSI300` 族 | `MKT_RET20/MKT_MA60/MKT_VOL20_PCT` 的 000300.SH/000905/000852/399006 多指数版 | 宽基指数日线（tushare index_daily 可直接补） | A,E |
| 14 | `STK_LIMIT` 精确版 | 用 stk_limit 表替换 LIMITCNT/LIMITDOWN/DIST_LIMIT 的 pct_chg 阈值近似 | 涨跌停价格表 | A |
| 15 | `LINK_MOM` | 同概念/共提及股票组合 20 日平均收益 | 概念板块历史成分或新闻共提及数据（成本高、前视风险大，最低优先） | A |

---

## 5.【不可算】清单（7 条）

本地数据原理上算不了，引擎**不得**出现这些特征名；若未来购入数据，先回本表降级再实现。

| # | 规范名 | 缺什么 | 来源 |
|---|---|---|---|
| 1 | `CPV_MIN`（分钟量价相关三维） | 分钟线 | A |
| 2 | `RSV_SEMIVAR`（RS+/RS- 半方差分解） | 分钟线 | A |
| 3 | `SMART`（聪明钱） | 分钟线 | A |
| 4 | `APM`（日内行为模式） | 分钟线 | A |
| 5 | `TAIL_RET`（尾盘收益占比） | 分钟线 | A |
| 6 | `MFLOW`（主力净流入占比） | 分单资金流 | A |
| 7 | `ORDER_FLOW` 类（盘口/逐笔/撤单/真实分钟量比/日内锚定 VWAP） | L2/tick/分钟 | A,C |

---

## 6. 特征工厂层（G 层，原称"程序化生成层"）

社区收割的 ~1300 指标本质是"字段×变换×窗口×算子"笛卡尔积。
G 层**不作为逐条收割特征**，而是本引擎的程序化生成层，统一生成、统一审计、统一筛查。

### 6.1 生成规则

- **基础字段集**（14 个）：`Õ,H̃,L̃,C̃,V,log(V+1),A,VWAP(=A/V),R,RET_ON,RET_ID,H/L,CF,R_idx`（指数仅用于双序列算子）。
- **变换/算子集**（24 个，对齐 qlib ops.py 与 Alpha191 语法）：`Ref, Delta, Mean, Sum, Std, Var, Skew, Kurt, Max, Min, Med, Quant(0.2/0.8), TsRank, IdxMax, IdxMin, Slope, Rsquare, Resi, EMA, WMA/DecayLinear, Count(cond), Corr(双序列), Cov(双序列)`。
- **窗口集**：`{3,5,10,20,30,60}`（5-30 日持仓主用 5/10/20/30；60 为长尾）。
- **后处理集**：`/C̃ 归一、-1× 反向、cs_rank（横截面）、zscore_ts`。
- K 线形态 9 式（qlib kbar：KMID/KLEN/KMID2/KUP/KUP2/KLOW/KLOW2/KSFT/KSFT2）与价格快照 4 式（Alpha158 price 组）作为固定小组随生成层产出。
- 命名沿用 qlib 惯例 `前缀+窗口`（如 `ROC20`、`CORR10`），每条生成特征把表达式字符串写入特征注册表，防止口径漂移（C 报告 §14）。

### 6.2 预计量级

- 主体：14 字段 × 约 20 个适用算子 × 6 窗口 ≈ 1,400-1,700 列；
- 双序列 Corr/Cov（约 8 对字段 × 6 窗口）≈ 50-100 列；
- K 线/快照固定组 13 列；
- 横截面 `cs_rank` 变体为可选第二层（开启后列数翻倍）。
- **单次生成总量预计 1,500-2,500 列**；与 P0/P1 手写特征去重后进入筛查。

### 6.3 筛查承接（机器生成特征一律无先验）

G 层特征不继承任何文献/历史证据，全部走"稳定性三关"：
1. **泄漏结构关**：与 P0/P1 相同的截断历史重算对拍（第 7 章），逐列通过。
2. **分年稳定关**：按 eval-protocol-v2 快循环，分年 Rank IC 符号一致性与 null95 通过率；G 层特征不得因单年表现入选。
3. **共线性关**：与 P0/P1 库及彼此做共线性聚类，每簇只留代表；被 P0 手写特征覆盖的生成特征直接淘汰。
通过三关的 G 层特征按"配置为行"的固定汇报格式出表，与 P0/P1 同口径评估。

---

## 7. 泄漏防线（引擎硬性要求）

### 7.1 列名黑名单（CI 硬卡点）

特征矩阵任何列命中以下正则即构建失败：

```
^stop_loss_            # V3 实锤泄漏族（7+7 列，未来交易模拟）
^future_               # 未来收益/未来日期
^next_                 # 回测执行列 next_open/high/low/close
^label                 # 标签命名空间
^mfr_                  # V5 标签族
^cur_return$|^max_forward_return$|^open_exec_return
^rank_future_|^rank_open_exec_
^ret_h\d+$             # divergence_lab 固定窗标签
^hit_N                 # 狙击标签
^mfe_|^mae_|^tmfe|^tmae # 最大有利/不利变动及时间
^dyn_                  # 动态窗标签
^entry_date$           # 回测执行列
^rank_                 # 该前缀整体保留给排名标签；特征横截面变体一律用后缀 _csrank/_csz
```

教训备案（L 报告 §1.1）：`stop_loss_return_*` 不含 future/next 关键词曾漏过关键词扫描，因此黑名单只是兜底，**真正的防线是 7.2 的结构测试**。

### 7.2 结构性测试（每个特征的准出门槛）

1. **截断历史重算对拍**：每个特征（含 G 层全部生成列）抽样 ≥3 只真实股票 × ≥50 个日期，将输入数据截断至 T 重算，与全历史计算在 T 处的值逐一比对（rtol≤1e-6，逐股隔离）。
   复用 `v3_pipeline/tests/test_divergence_lab.py` 的暴力逐日截断对拍框架。
2. **函数字段白名单**：特征注册表记录每条特征的输入字段；出现白名单（1.1 节）之外的字段即失败。
3. **标签命名空间隔离**：标签只能由 label 模块（label_candidates/ranking_labels/divergence_lab 标签族）产出，特征引擎只读不写；特征选择配置必须显式 exclude 黑名单正则。
4. **样本过滤禁令**：任何基于 T 之后数据的可成交性/过滤逻辑（V1 `is_valid_row` 偷看 T+1 low 的教训）不得出现在特征与样本构建层；可成交性由 label NaN 处理。

### 7.3 可疑 6 条处置决定（L 报告 §4 逐条）

| # | 可疑项 | 处置 |
|---|---|---|
| 1 | `hs_*`/`gq_*`/`hc_*` 旧三指数列 | **弃用**。指数对应关系不可考；市场特征一律按 sh/sz 双指数口径（P1 #45/#46）重算并显式命名。如需三指数，先补数据走 P2-13 |
| 2 | `volume_signal` 字符串事件字段 | **弃用字符串版**。重定义为数值 `VOL_CONFIRM ∈ {-1,0,1}`（放量/平/缩量）事件字段，归特征工厂层；引擎排除规则精确到列名，禁止 `.*_signal$` 后缀误伤（历史误伤过 `sh_volume_signal`） |
| 3 | v1 滑窗低点锚定（6 窗 step=3） | **废弃**。低点锚定只允许 v2 检测器 left3/right2 稳定锚或 divergence_lab fractal/zigzag；事件特征（P0 52-59 等）只接受这三套锚的输入 |
| 4 | `macd_percentile` 的 NaN/==50 口径 | **重定义**：改名 `DIF_PCTILE100` = DIF 在过去 100 值的 trailing 百分位，min_periods=60，明确 NaN 策略（不置 0），归特征工厂层 |
| 5 | `rank_volume`/`signed_vol_strength` 的 `hs_volume_ratio` fallback | **删除 fallback**，固定用个股 V；引擎禁止任何"列存在即替换语义"的隐式分支 |
| 6 | `pv_corr_10` 跨股边界屏蔽写法 | **重写**为标准 `groupby(ts_code).rolling(10).corr()`，min_periods=10，保留 NaN 交模型处理（即 P0 #21 `CPV10` 的实现口径） |

### 7.4 工程纪律（L 报告 §6 固化）

1. 全历史逐股计算 → 末端按日期切片；禁止 v2 的"先截 100 天再算"。
2. 命名清理：`boxcox_atr` 改名 `LOG_ATR`；两处 `price_volume_divergence` 重名不得复活（语义分别由 `CPV*` 与特征工厂层覆盖）；`macd_signal_cross` 并入 `MACD_GOLDEN_CROSS`。
3. `_csrank`/`_csz`/`cs_n` 横截面后处理为引擎通用机制，对全部数值特征自动派生；仅按 trade_date 分组，无时间方向泄漏。
4. 事件特征与检测器版本绑定：注册表记录低点锚定方法与参数指纹，沿用 v2 的 `compute_cache_fingerprint` 机制。

---

## 8. 汇总

| 清单 | 条数 |
|---|---|
| 【可算】P0 | 60 |
| 【可算】P1 | 81 |
| 【需补数据】P2 | 15 |
| 【不可算】 | 7 |
| 特征工厂层 G | 1 层（预计 1,500-2,500 列，走稳定性三关） |

去重说明：四源原始条目（学术 70 + 事件 50 + 历史 114 安全 + 社区 ~150 算子体系）按"经济含义+公式等价"合并为 141 条可算命名特征 + 15 条需补 + 7 条不可算 + 1 个生成层；社区 Alpha158/191/101/TA-Lib 的逐条公式不单列，由 G 层语法覆盖。
