# 本地数据与特征代码盘点（feature harvest 前置）

盘点日期：2026-09-01。
目的：为大规模特征工程划清可计算边界。
范围：只读盘点，覆盖原始数据、背离事件池、现有特征代码、计算基建、数据缺口。

---

## 1. 原始数据

### 1.1 股票日线（唯一的大规模本地数据）

路径：`stock_data/daily/*.parquet`，每股一文件（pyarrow/snappy），共 5891 个文件、约 833 MB、1733.7 万行。

拉取方式：`scripts/fetch_daily_data.py` 经 tinyshare `pro.daily(ts_code, start_date, end_date)` 拉取，**未传 adj 参数，即不复权原始价**。
本地无 adj_factor，复权因子算不了。

股票文件 schema（tushare daily 原样，11 列）：

| 列 | 类型 | 说明 |
|---|---|---|
| ts_code | str | 如 000001.SZ |
| trade_date | datetime64[us] | 交易日 |
| open / high / low / close | float64 | OHLC，不复权 |
| pre_close | float64 | 前收盘（可算真实跳空） |
| change / pct_chg | float64 | 涨跌额 / 涨跌幅(%) |
| vol | float64 | 成交量（手） |
| amount | float64 | 成交额（千元） |

**没有**：换手率、量比、市值、PE/PB（无 daily_basic），无任何复权因子。

覆盖范围：
- 全局 1990-12-19 → 2026-08-31。
- 按市场后缀：SZ 3085、SH 2462、BJ 344（含 2 个指数文件，见下；北交所 344 只混在其中，特征工程需决定是否剔除）。
- universe_latest.parquet 记录 5892 只（在市 + 退市 341 只），字段仅 ts_code / name / list_date / delist_date。
- 5553 个文件更新至 2026-08-25 之后；338 个文件止于 2026-08-01 之前（退市股）。
- 文件行数中位数约 2943 行；上市中位日期 2015-12。
- 注意：`000852.SZ`（石化机械）、`000905.SZ`（厦门港务）是真股票，不是中证指数。

### 1.2 指数数据（仅 2 只，且 schema 不同）

`stock_data/daily/000001.SH.parquet`（上证指数）、`399001.SZ.parquet`（深证成指），由 `scripts/fetch_index_data.py` 经 `pro.index_daily` 拉取。
schema 与个股不同：`trade_date` 为字符串 YYYYMMDD，仅 OHLC + volume，**无 amount、pre_close、pct_chg、ts_code**。
各 8000 行，1993 → 2026-08-31。
无 CSI300/500/1000、无行业指数、无其他资产类别。

### 1.3 不存在的本地数据

行业分类、财务数据、北向/资金流、分钟线、盘口/逐笔、涨跌停价格表、融资融券、龙虎榜、公告舆情：本地一概没有，代码中也没有任何接口调用落盘过这些数据。
`data/`、`stock_data/csv/`、`stock_data/st_filter/`、`real_feature_data_daily/`、`logs/`、`models/`、`output/` 均为空目录。
`real_trading_data/` 只有实盘组合 CSV（与特征工程无关）。
仓库瘦身后 V2 的 626 维特征缓存（feature_cache_all.parquet 等）已删除，仅存指纹文件。

---

## 2. 背离事件与标签（两个主力池）

位置：`v3_pipeline/reports/divergence_lab/m_scan/{pool}/`，各含 events.parquet / labels.parquet / stats.json。
配置：`v3_pipeline/configs/divergence_lab/m_scan/*.json`。

### events.parquet 列（两池一致）

`event_id`(i64)、`ts_code`(str)、`sig_idx`(i32，股内行号)、`date`(datetime64[ms]，信号日)、`low_date`/`prev_low_date`(i32，日序)、`compare_rank`(i8，与前第几名低点背离)、`formation`(i32，两低点间隔根数)、`regime`(str：up/down/sideways/unknown)、`above_ma200`(bool)。

### labels.parquet 列（两池一致）

`group`(str：div=事件 / c1=同股随机非事件日 / c2=同日随机非事件股) + 6 个 float32 标签列：`ret_h10`、`ret_h30`（close_T 入场固定周期收益，pct_chg 链口径）、`hit_N20_k2.0`、`hit_N20_k3.0`、`hit_N40_k2.0`、`hit_N40_k3.0`（狙击标签：T+1 开盘入场，N 日内 high 触及 开盘价+k×ATR14(信号日) 则 1）。
div 组与 events.parquet 按位对齐（architecture_race.py 有断言）。

### 规模

| 池 | 事件数 | 事件股数 | 信号日范围 | labels 行数 (div/c1/c2) | regime 分布 |
|---|---|---|---|---|---|
| m_fractal15_full（主池） | 8,158 | 4,039 | 1993-04-26 → **2026-05-26** | 49,413 (8158/33097/8158) | 全部 up（配置过滤 regime=up） |
| m_zigzag05_nofilter（备池） | 37,012 | 5,512 | 1992-10-12 → 2026-08-31 | 111,036 (37012×3) | down 14597 / sideways 13090 / up 9325 |

池配置要点（两池共有）：indicator=DIF、min_change=0.001、below_zero=true、min_decline=0.08、lookback=2、volume_confirm=true、entry=close_T。
差异：主池 fractal(order=15, min_sep=20) + regime=up 过滤；备池 zigzag(pct=0.05, min_sep=5) + 无过滤。
注意：主池事件止于 2026-05-26（regime=up 过滤 + 末端行情阶段所致，非数据截断）。
其他 run 池：`divergence_lab/`（w_/d_/smoke 共 43 个）、`divergence_lab_c/`（c_* 对照 27 个）、`divergence_event_study/events.parquet`（V1 legacy 池）。

---

## 3. 现有特征代码

### 3.1 architecture_race.py 的 18 维特征（`/home/karl/repos/personal/stock_qt_nd/v3_pipeline/scripts/architecture_race.py`）

全部严格因果（只用信号日 T 及之前）。
来源分两类的：`compare_rank`/`formation`/`above_ma200`/`regime_code` 直接取 events.parquet；其余由 `_stock_features` 按股补算（复制 divergence_lab 数据口径）。

| # | 列名 | 口径 |
|---|---|---|
| 1 | compare_rank | 与前第几名低点形成背离（1 或 2，events） |
| 2 | formation | 当前低点与前低点的间隔根数（events） |
| 3 | confirm_lag | sig_idx − low_idx，低点到信号确认日的滞后根数 |
| 4 | price_decline | close[low]/close[prev_low] − 1，两低点间价格跌幅 |
| 5 | dif_lift_atr | (DIF[low] − DIF[prev_low]) / ATR14[sig]，指标抬升幅度（ATR 单位） |
| 6 | price_drop_atr | (close[prev_low] − close[low]) / ATR14[sig]，价格下跌幅度（ATR 单位） |
| 7 | dif_sig | 信号日 MACD DIF(12,26,9) 值 |
| 8 | dif_low | 低点日 DIF 值 |
| 9 | atr_pct | ATR14[sig] / close[sig]，相对波动率 |
| 10 | log_close | log(close[sig])，价格水平 |
| 11 | above_ma200 | close[sig] > MA200（events，0/1） |
| 12 | ma200_ratio | close[sig]/MA200[sig] − 1 |
| 13 | regime_code | 市场阶段编码（unknown=-1/sideways=0/up=1/down=2，events regime 映射） |
| 14 | vol_shrink | vol[low] / vol[prev_low]，低点间量能比 |
| 15 | vol_ratio | vol[sig] / mean(vol[sig−20..sig−1])，信号日量比（20 日） |
| 16 | ret_5 | 信号日前 5 日收益（close 口径） |
| 17 | ret_10 | 前 10 日收益 |
| 18 | ret_20 | 前 20 日收益 |

附属：规则基线 `rule_score = dif_lift_atr + price_drop_atr`（背离强度，ATR 单位）。
辅助补算 `ret20`（T+1 开盘入、T+21 收盘出，与狙击标签同入场口径）用于 PF/盈亏比。

### 3.2 divergence_lab.py 已实现的衍生量（`/home/karl/repos/personal/stock_qt_nd/v3_pipeline/scripts/divergence_lab.py`）

低点识别（三种，全部因果）：`fractal_lows`（order=k 严格分形，信号日=低点+k）、`zigzag_lows`（流式状态机，反弹达 pct 当日确认前低，numba）、`legacy`（V1 生产截断语义复刻）、`apply_min_sep`（低点最小间隔过滤）。
背离判定条件：价格新低 + 指标抬高（min_change）、可选 below_zero（两低点 DIF 均 <0）、min_decline（次低最小跌幅）、volume_confirm（次低缩量）、lookback（回看低点数）、multi=3（连续三低点多重背离）。
事件衍生量：compare_rank、formation、sig/low/prev 索引。
指标链：talib MACD(12,26,9) 的 DIF/HIST、ATR14、SMA200、`cf` = pct_chg 链式累乘收益因子（规避除权跳变的收益口径）。
市场阶段：`build_market_regime` 全样本等权日收益代理指数，120 日滚动收益 ±10% 分 up/down/sideways/unknown。
标签族：`ret_h{h}` 固定周期（close_T 或 open_T1 入场）、`dyn`（h=c×formation 封顶）、`hit_N{N}_k{k}` 狙击、mfe_h{w}（最大有利变动）。
双对照：C1 同股随机非事件日（seed+crc32(symbol) 确定性）、C2 同日随机非事件股。
统计：胜率/均值/中位数/超额、Welch t、Mann-Whitney，分 explore/validate、分年度、分 regime。

### 3.3 src/divergence_detector.py（V1 遗产）与 divergence_event_study.py

V1 检测器（`src/divergence_detector.py`，v2 同构）：6-bar 滑窗步长 3 局部低点；派生量 price_decline_pct、macd_increase_pct、divergence_strength（0.4×价格强度+0.6×MACD 强度，各按 0.1 封顶）、formation_period、is_quick_divergence、volume_signal（bullish/bearish/neutral）、compare_rank、price_macd_ratio、divergence_magnitude、confirmation_score。
`v3_pipeline/scripts/divergence_event_study.py::simulate_events_idx`：V1 截断语义的解析因果形式（divergence_lab 的 legacy 方法直接复用，MIN_LEN=100）。
遗留大特征库：`src/feature_pipeline.py`/`feature_pipeline_v2.py`（V2 的 626 维，文档见根目录 features.md / feature_readme.md，含 MACD/RSI/MA/布林/OBV/KDJ/K线形态/大盘特征等）——代码在但特征缓存已删除，重建成本未评估。

### 3.4 其他相关模块

`v3_pipeline/src/label_candidates.py`：V5 标签族纯函数（cur=收盘到收盘、open_exec=T+1 开盘执行、mfr=窗口内最大有利收益）。
`v3_pipeline/src/ranking_labels.py`：横截面百分位排名标签。

---

## 4. 计算基建

数据加载方式（divergence_lab.py `run`/`load_stock`）：glob 全宇宙 parquet → ProcessPoolExecutor（默认 8 workers，chunksize=16）逐股读入 → dropna(close)/dedup(trade_date)/sort → talib 指标 → 以 f32/i32 紧凑字典常驻内存。
全宇宙单配置端到端耗时约 9–10 秒（两池 stats.json 的 elapsed_s 实测值，含事件检测+标签+双对照+统计），可支撑大规模参数扫描。
architecture_race 特征补算为 16 workers 按股并行，18 维对全事件几分钟内完成。

可复用的因果对拍测试：`v3_pipeline/tests/test_divergence_lab.py` —— fractal/zigzag 用暴力逐日截断法对 3 只真实股票（600519.SH/000001.SZ/000002.SZ）全量对拍 0 mismatch；legacy 与 V1 因果模拟全量对拍；合成数据覆盖背离规则与全部标签族。
另有 `test_label_candidates.py`、`test_ranking_labels.py`；根 `tests/` 下有 test_regression_divergence.py、test_universe.py 等。

环境：Python 3.11.8、pandas **3.0.5**（新大版本，老代码兼容性需留意）、numpy 2.4.6、talib 0.7.1、lightgbm 4.7.0、numba 0.67.0、scipy 1.17.1、sklearn 1.9.0、pyarrow 25.0.1。
硬件：62 GB RAM、28 核、NVMe SSD（余量 420 G）。
内存量级线索：全宇宙 f32 常驻约 17.3M 行 × ~10 个数组 ≈ 1 GB 量级，62 GB 内存余量充足，全宇宙横截面特征矩阵（5891 股 × 8500 交易日 × float32 ≈ 200 GB/列宽级）需按特征分块或稀疏存储，不能一次性全宽展开。

---

## 5. 缺口清单（对照学术/社区常见短周期因子）

### 因缺数据算不了的因子类别

| 因子类别 | 缺什么 |
|---|---|
| 日内/分钟因子（开盘半小时动量、尾盘成交占比、日内路径、VWAP 偏离、隔夜-日内收益分解） | 无分钟数据（隔夜收益可用 open/pre_close 近似一半） |
| 盘口/逐笔/订单流（买卖压、撤单、大单净流入） | 无 L2/tick |
| 资金流因子（主力/超大单净流入、北向持股变动） | 无 moneyflow/hsgt |
| 换手率、真实量比、流通市值/总市值（size 因子）、PE/PB/PS 估值 | 无 daily_basic；换手需流通股本，市值需总股本，本地均无 |
| 行业因子（行业动量、行业中性化、行业轮动） | 无行业分类映射（申万/中信） |
| 基本面因子（ROE、营收/利润增速、毛利率、杠杆） | 无财务数据 |
| 精确涨跌停/一字板标记 | 无 stk_limit 价格表；可用 OHLC+pct_chg+名称近似但非事时精确 |
| 事时 ST/退市风险标记 | universe 只有当前 name 与 list/delist 日期，无历史 ST 状态 |
| 精确停复牌标记 | 只能从缺失交易日推断 |
| 除权感知的价格水平因子 | 无 adj_factor：价格为不复权原始价，收益用 pct_chg 链已规避，但 log_close、52 周高点距离、价格分位等**价格水平类因子在除权股上失真**（重要 caveat） |
| 宽基/行业指数相对强弱（CSI300/500/1000） | 本地仅上证 000001.SH、深证成指 399001.SZ（且 schema 不同、无 amount） |
| 融资融券、龙虎榜、公告/舆情/分析师预期 | 无 |

### 本地可算的因子类别（基于 OHLCV + amount + pre_close + pct_chg）

动量/反转（任意窗口）、波动率与 ATR 族、RSI/MACD/KDJ/布林等全量技术指标（talib 现成）、振幅/上下影线/实体形态、跳空（pre_close）、量能趋势/放量缩量/量价相关、Amihud 非流动性（|ret|/amount，amount 可用）、相对两只指数的强弱与 beta、市场阶段代理（已有 regime 基建）、事件结构衍生（背离强度、形成期、确认滞后等 18 维的任意扩展）、隔夜收益（open/pre_close−1）。
横截面排名/中性化基建可复用 ranking_labels.py。
