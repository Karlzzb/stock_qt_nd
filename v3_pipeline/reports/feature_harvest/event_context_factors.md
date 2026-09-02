# MACD 底背离事件语境特征收割清单

生成日期：2026-09-01。
用途：在背离信号池内区分"会涨的背离"与"假背离"，特征全部为**信号日 T 收盘后可算**的语境变量。
标签约定：T+1 开盘买入，20 个交易日内 +2×ATR(14) 触达为命中。

## 记号约定

- 日线 OHLCV：`O/H/L/C/V/AMT`（成交额），全部为**前复权**口径。
- MACD(12,26,9)：`DIF = EMA12(C) - EMA26(C)`，`DEA = EMA9(DIF)`，`HIST = 2×(DIF - DEA)`（国内口径乘 2，是否乘 2 不影响任何比值类特征）。
- 背离结构：第一价格低点 `L1`（低点日索引 `i1`），第二价格低点 `L2`（低点日索引 `i2`，`i2 > i1`），`L[i2] < L[i1]` 且 `DIF[i2] > DIF[i1]`；确认日 `T ≥ i2`。
- `ATR14`：Wilder 14 日平均真实波幅；`ATRp = ATR14 / C`（相对波动率）。
- `MA_n`：n 日简单均线；`EMA_n`：n 日指数均线；`LLV_n(X)/HHV_n(X)`：n 日最低/最高；`σ_n(r)`：日收益 n 日标准差。
- `rank_cs(x)`：当日全市场横截面分位（0~1）；`pct_ts(x, n)`：自身过去 n 日时间序列分位。
- 硬约束：所有特征仅用 `t ≤ T` 的数据计算，T 收盘后可得；标注「需补充数据」的特征依赖仓库当前没有的外部数据，已单独注明。

## 一、背离事件本身的微观结构（14 个）

| # | 特征名 | 精确公式 | 方向假设 | 证据/来源 |
|---|--------|----------|----------|-----------|
| A1 | `div_span_bars` 背离跨度 | `i2 - i1`（交易日数） | 中等跨度（10~40 根）优于过近（噪音）与过远（结构失效）；作为连续变量交给模型 | 中文社区统计文章普遍讨论背离时间间隔与有效性关系：[技术指标背离几次有效？MACD顶底背离的统计规律](https://blog.csdn.net/liuyun12139/article/details/148363647)（无法抓取正文，仅作方向性参考） |
| A2 | `div_span_vs_cycle` 跨度相对性 | `(i2 - i1) / (自 T 起最近一次 DIF 完整正负循环的时长)` | 跨过至少一个完整绿柱周期的背离更可靠 | 同上，背离定义本身要求两低点间 DIF 回摆 |
| A3 | `div_price_newlow_depth` 新低幅度 | `L[i2]/L[i1] - 1`（负值） | 浅新低（-1%~-5%）优于深新低（恐慌未出清） | 实战共识："价创新低幅度小、指标明显抬高"才是强背离，见[知乎:MACD底背离成功率讨论](https://www.zhihu.com/question/589509227)、[知乎:MACD背离正确率](https://www.zhihu.com/question/20665577) |
| A4 | `div_dif_lift` DIF 抬升幅度 | `(DIF[i2] - DIF[i1]) / (C[i2] × ATRp[i2])`，即按波动率归一的 DIF 抬升 | 抬升越大，动能背离越强 | 同上；背离幅度量化是中文公式社区的标配做法，见[Macd指标背离用法大全+选股公式](http://www.360doc.com/content/23/0609/22/3534277_1084137162.shtml) |
| A5 | `div_dif_slope_between` 两低点间 DIF 斜率 | `(DIF[i2] - DIF[i1]) / (i2 - i1)`，再除以 `C[i2]×ATRp[i2]` 归一 | 斜率越陡（抬升快）越好 | 同 A4 |
| A6 | `div_zero_axis_depth` 零轴位置 | `max(DIF[T], DEA[T]) / (C[T] × ATRp[T])`（负值=零轴下深度） | 已知基线过滤「零轴下」有效，深度本身作为连续变量可能非单调：过深=趋势性崩坏，适中最好 | 基线事实 + [零轴上方金叉才靠谱](https://post.smzdm.com/p/apq2m2ow)（反向佐证零轴位置承载信息） |
| A7 | `div_dif_level_l1` 首低点 DIF 深度 | `DIF[i1] / (C[i1] × ATRp[i1])` | 首低点越深，说明前面下跌动能越充分释放，二次探底失败的概率越低 | 中文实战："DIF 越深，背离越值钱"，见[MACD底背离找准最佳买点技巧](http://mp.weixin.qq.com/s?__biz=Mzg4NTUxNjY0MQ==&mid=2247504627&idx=1&sn=71510b37da6c8952f25986bc9163cdad)（无法抓取正文，方向性参考） |
| A8 | `div_count_120` 第几次背离 | 过去 120 个交易日内（含本次）满足同向底背离定义的次数，1/2/3+ | 二次、三次背离成功率高于首次（空头动能逐级衰竭） | [技术指标背离几次有效？统计规律](https://blog.csdn.net/liuyun12139/article/details/148363647)；[MACD二次底背离买入机会分析](https://www.55188.com/thread-24623527-1-1.html)；[二次金叉配合背离提升成功率](https://m.163.com/dy/article/GTTHVPJ605527FD0.html) |
| A9 | `div_hist_area_shrink` 柱线负面积收缩 | `S_k = Σ HIST[t]`（HIST<0 部分，第 k 个低点前最近的绿柱区间）；特征 = `S_2 / S_1`（|S₂|<|S₁| 时 <1） | 越小（收缩越狠）越好：空方动能量级衰减 | [Macd指标背离用法大全](http://www.360doc.com/content/24/0401/22/25547610_1119159050.shtml)中"绿柱面积背离"的量化版本 |
| A10 | `div_hist_trough_shrink` 柱线极值收缩 | `min(HIST 于 i2 附近绿柱区) / min(HIST 于 i1 附近绿柱区)`（负值之比，<1 为收缩） | 同 A9，对单根异常柱更敏感 | 同 A9 |
| A11 | `div_golden_cross_state` 金叉确认状态 | 三值：`1[DIF[T]>DEA[T] 且 DIF[T-1]≤DEA[T-1]]`（T 日金叉）、`1[T 日前 5 日内已金叉]`、`0`（未金叉） | 已金叉 > T 日金叉 > 未金叉；金叉=背离的事后确认 | [MACD二次金叉买入法配合背离](https://m.163.com/dy/article/GTTHVPJ605527FD0.html)；[掌握MACD二次金叉与底背离选股公式](https://gongshi.zaixianjisuan.com/gongshi/zhang-wo-macder-ci-jin-cha-yu-di-bei-li-xuan-gu-gong-shi-ti-gao-tou-zi-cheng-gong-lv.html) |
| A12 | `rebound_from_l2` 初始反弹力度 | `(C[T] - L[i2]) / ATR14[T]` | 背离后已有 1~2×ATR 的初始抬升=资金承接实锤；完全无抬升=死猫 | 学术侧证据：MACD 信号需配合确认条件才有效，[arXiv:2206.12282](https://arxiv.org/pdf/2206.12282/1000)；中文侧见[底背离选股战法](https://www.sohu.com/a/224860994_543200)（无法抓取正文，方向性参考） |
| A13 | `days_since_l2` 确认时滞 | `T - i2` | 时滞过长（>10 日）说明背离迟迟未被资金确认，质量存疑 | 与 A11/A12 同源的确认逻辑 |
| A14 | `rebound_day_T` 信号日强度 | `(C[T]-O[T])/ATR14[T]` 与 `pct_chg[T]/100/ATRp[T]` 取前者为主 | T 日为放量实体阳线=当日即有承接 | [黄金坑战法回测](http://mp.weixin.qq.com/s?__biz=MzU3MDcwOTE4Nw==&mid=2247501486&idx=2&sn=32e89e698e40da714609675c2dd5f0a8)类文章均强调反转 K 线实体强度（无法抓取正文，方向性参考） |

## 二、个股状态（14 个）

| # | 特征名 | 精确公式 | 方向假设 | 证据/来源 |
|---|--------|----------|----------|-----------|
| B1 | `pos_52w` 52 周区间位置 | `(C[T] - LLV_252(L)) / (HHV_252(H) - LLV_252(L))` | 非单调：贴近 52 周低点（<0.1）可能是接飞刀也可能是黄金坑；交给模型与 A 组交互 | George & Hwang (2004) 52 周高点锚定动量研究，综述见[Momentum: 30 years after Jegadeesh & Titman](https://link.springer.com/article/10.1007/s11408-022-00417-8)与[52-Week High and Momentum Investing](https://edubirdie.com/docs/university-of-houston/fina-4330-corporate-finance/111531-the-52-week-high-and-momentum-investing) |
| B2 | `dist_52w_low` 距 52 周低点 | `C[T] / LLV_252(L) - 1` | 距低点 5%~20% 的"已离开底部但未走远"最优（与 B1 互补的绝对量纲） | Bulkowski 统计：锤子线在年内低点三分之一区域内表现最好，[Bulkowski on the Hammer](https://thepatternsite.com/Hammer.html) |
| B3 | `dist_52w_high` 距 52 周高点 | `C[T] / HHV_252(H) - 1`（负值=回撤深度） | 深回撤股反弹空间大但趋势破坏重；与 B1 共线但截距不同，可并用 | [Maximum drawdown, recovery, and momentum (arXiv:1403.8125)](https://arxiv.org/pdf/1403.8125)：最大回撤与恢复路径对未来收益有预测力 |
| B4 | `trend_slope_120` 长期趋势斜率 | 对 `ln(C)` 过去 120 日做 OLS，取斜率 ×120（年化感知尺度），再除以 `σ_120(r)×√120` 得 t 统计量版本 | 基线已证"仅上涨段"过滤有效；长期斜率向上 + 短期背离=回调买点的经典结构 | 基线事实（已知 65-71% 命中池靠"仅上涨段"）+ [行业动量轮动研报](https://zhuanlan.zhihu.com/p/29986910934)中趋势过滤思路 |
| B5 | `ma_stack_state` 均线排列 | `sign(C[T]-MA60)`、`sign(MA60-MA120)`、`sign(MA120-MA250)` 三个哑变量 | 股价在长期均线簇上方回踩的背离质量最高 | 中文实战共识，见[MACD实战：3招捕捉主升浪](http://mp.weixin.qq.com/s?__biz=MzI0MTQ1NDE5NA==&mid=2247484067&idx=1&sn=12678bc86c46b5330f672ff7c2d3b6e9) |
| B6 | `atrp_level` 波动率水平 | `ATRp[T]`，及 `pct_ts(ATRp, 250)` | 背离发生在自身高波动分位=恐慌出清中；低波动分位=阴跌，两者结局不同 | [Bollinger Bands: Squeeze, BandWidth and Volatility Regimes](https://volity.io/forex/bollinger-bands/) |
| B7 | `vol_contraction_60` 波动收缩比 | `ATRp[T] / ATRp[T-60]` | <0.8 为收缩：下跌末段波动收敛=抛压衰竭；>1.2 为扩张：下跌加速中 | 挤压-扩张文献：[LuxAlgo: Squeeze then Surge](https://www.luxalgo.com/blog/bollinger-bands-strategy-squeeze-then-surge/)、[QuantCase: BB Squeeze NSE](https://www.quantscase.com/blog/bollinger-band-squeeze-nse-strategy) |
| B8 | `bbw_pctile` 布林带宽分位 | `BBW = (UB-LB)/MA20`（20 日、2σ），取 `pct_ts(BBW, 250)` | 低分位（<0.2）=挤压末端，方向待定但幅度可期；与背离共振时向上概率大 | [PyQuantLab: RSI Filtered BB Squeeze](https://pyquantlab.com/article.php?file=RSI%20Filtered%20Bollinger%20Band%20Squeeze%20Strategy.html)（挤压+RSI 过滤的正收益回测）；注意挤压本身不预测方向，[Volity](https://volity.io/forex/bollinger-bands/) |
| B9 | `ret_20_pctile` 近期跌幅分位 | `r_20 = C[T]/C[T-21]-1`，取 `rank_cs(r_20)` 与 `pct_ts(r_20, 250)` 两个版本 | 横截面跌幅越深（分位越低）短期反转期望越强（A 股短期反转效应显著） | [个股反转策略与行业动量策略——A股实证](https://www.sinoss.net/c/2017-05-23/558883.shtml)（短期反转月均超额 0.57%）；[BigQuant 动量因子实证](https://bigquant.com/wiki/doc/vmpoW4sE1e)（3 月回看 IC=-0.032 反转） |
| B10 | `drawdown_60` 60 日回撤 | `C[T] / HHV_60(H) - 1` | 基线「跌幅≥8%」的连续化版本；回撤深度与反弹空间正相关但过深=基本面崩坏 | 基线事实 + [BIAS乖离率超跌反弹批量回测](https://www.zpyztech.com/bias-oversold-rebound-batch-backtest-20260731/) |
| B11 | `gap_down_T` 跳空缺口 | `O[T]/C[T-1] - 1`；另取过去 20 日最大向下跳空 | 衰竭性跳空（下跌末段缺口+当日收复）是经典底部信号；T 日大幅低开未收复=恐慌延续 | 衰竭缺口为经典技术文献概念；中文复盘框架见[大盘环境复盘框架](https://ag.yueniuzq.com/market-review/build-market-environment-review-framework/) |
| B12 | `vol_dryup_5_60` 量能枯竭比 | `MA_5(V)[T] / MA_60(V)[T]` | 基线「缩量」的连续化；<0.6 为明显枯竭，抛压衰竭 | 基线事实 + [叩富网:放量上涨与缩量下跌的含义](https://licai.cofool.com/user/guide_view_3377003.html)（缩量下跌=抛压减弱）；[量价背离百度百科](https://baike.baidu.com/item/%E9%87%8F%E4%BB%B7%E8%83%8C%E7%A6%BB/4866454) |
| B13 | `vol_dryup_extreme` 极端地量哑变量 | `1[MA_5(V)[T] < quantile_0.1(MA_5(V), 过去250日)]` | 地量见地价：极端缩量后变盘概率上升，需配合背离方向过滤 | [叩富网:量价背离](https://licai.cofool.com/ask/qa_7448683.html)；[约投顾:成交量剧增与缩量横盘](https://ag.yueniuzq.com/stock/volume-price-relationship-shrink-surge-trend/) |
| B14 | `amt_shrink_vs_peak` 成交额相对峰值萎缩 | `MA_5(AMT)[T] / max(MA_5(AMT), 过去120日)` | 从峰值萎缩 >70% 说明交投冰封，筹码锁定，反弹时抛压轻 | 地量文献同上；成交额比成交量更抗股本结构干扰 |

## 三、市场环境（9 个）

数据基础核查（2026-09-01）：`stock_data/daily/` 内有 `000001.SH`（上证综指，1993 起 8000 行）与 `399001.SZ`（深证成指）；`000905.SZ` 存在但来源需核实；沪深300（`000300.SH`）与 `399006.SZ` 缺失，需补下载。
宽度类特征可用 `stock_data/daily/` 内 5891 只股票的 universe 自行横截面计算，无需外部数据。

| # | 特征名 | 精确公式 | 方向假设 | 证据/来源 |
|---|--------|----------|----------|-----------|
| C1 | `idx_trend_ma60` 大盘趋势 | `C_idx[T]/MA60_idx[T] - 1`（指数默认上证综指，备选 000905 中证500） | 指数在 MA60 上方时背离成功率高；深度熊市中背离层层失效 | 多指标组合在趋势市优于震荡市：[QuantifiedStrategies: MACD accuracy](https://www.quantifiedstrategies.com/what-is-the-accuracy-of-macd-trading-strategies/)；[BigQuant 因子实证](https://bigquant.com/wiki/doc/vmpoW4sE1e)（因子有效性市场状态依赖） |
| C2 | `idx_ret_20` 大盘近期动量 | `C_idx[T]/C_idx[T-21] - 1` | 大盘 20 日急跌后的背离=恐慌共振底；大盘阴跌中=胜率平庸 | [阅牛:大盘环境复盘框架](https://ag.yueniuzq.com/market-review/build-market-environment-review-framework/)（指数+成交额+涨跌家数三维框架） |
| C3 | `idx_vol_state` 大盘波动状态 | `pct_ts(σ_20(r_idx), 250)` | 高波动分位=系统性风险释放期，假背离多；波动回落期反弹质量高 | [Volity: Volatility Regimes](https://volity.io/forex/bollinger-bands/) |
| C4 | `idx_drawdown_120` 大盘回撤 | `C_idx[T] / HHV_120(H_idx) - 1` | 大盘回撤 10%+ 后的个股背离属于"被错杀"型，反弹弹性大 | [Maximum drawdown, recovery, and momentum](https://arxiv.org/pdf/1403.8125) |
| C5 | `idx_rsi14` 大盘超卖度 | `RSI14(C_idx)[T]` | 指数 RSI<30 时个股背离共振成功率高（市场级超卖） | RSI 过滤提升 MACD 策略胜率：[arXiv:2206.12282](https://arxiv.org/pdf/2206.12282/1000)；[PyQuantLab RSI-Filtered Squeeze](https://pyquantlab.com/article.php?file=RSI%20Filtered%20Bollinger%20Band%20Squeeze%20Strategy.html) |
| C6 | `breadth_adv_ratio_5` 宽度：5 日上涨占比 | universe 内 `r_5 > 0` 的股票占比 | 宽度从极端低值（<0.2）回升=情绪冰点修复，背离反弹的黄金窗口 | [阅牛:用涨跌家数复盘市场情绪温度](https://ag.yueniuzq.com/market-review/how-to-use-advance-decline-count-to-review-sentime/)；[阅牛:涨跌家数判断次日情绪](https://ag.yueniuzq.com/market-review/judge-next-day-sentiment-from-advance-decline-data/) |
| C7 | `breadth_newlow_ratio` 创新低家数占比 | universe 内 `L[T] = LLV_250(L)` 的股票占比 | 占比从极值回落=抛压系统性衰竭（市场级底背离） | [阅牛:涨跌家数比变化感知情绪转折](https://ag.yueniuzq.com/market-review/review-advance-decline-ratio-sentiment-turning/)（底背离：指数新低但上涨家数不再新低） |
| C8 | `breadth_pct_above_ma20` 站上 20 日线占比 | universe 内 `C > MA20` 的股票占比 | <0.15 为极端冰点，此后背离反弹胜率与弹性双高 | [SkillsMP: A股市场宽度分析](https://skillsmp.com/pt/creators/aifinlab/finclaw/skills-a-share-market-breadth)；[阅牛:市场情绪指标量化](https://ag.yueniuzq.com/stock/market-sentiment-indicators-quantify-heat/) |
| C9 | `mkt_median_ret_20` 市场中位收益 | universe 内 `r_20` 的中位数 | 中位数深度为负时个股跌幅的"市场成分"高，背离反弹是 Beta 修复 | [阅牛:如何通过指数走势与个股表现判断市场真实强弱](https://ag.yueniuzq.com/market-review/judge-real-market-strength-via-index-stocks/)（中位数比均值更真实） |

## 四、行业/板块（4 个）

数据基础：仓库当前**无行业分类数据**，需补充申万一级/二级或中信行业映射（tushare `index_classify`/`stock_basic` 或 akshare 行业成分均可）。
补充后行业指数可用成分股等权/流通市值加权自行构建，保证 T 日可算、无供应商前视。

| # | 特征名 | 精确公式 | 方向假设 | 证据/来源 |
|---|--------|----------|----------|-----------|
| D1 | `ind_rel_strength_20` 行业相对强度 | `r_20(ind) - r_20(mkt)`，ind 为个股所属行业等权指数 | 行业强于大盘时的个股背离=强势行业内的回调买点；弱行业背离=逆水行舟 | 行业层面动量显著为正：[BigQuant 行业轮动"长短共振"](https://bigquant.com/square/paper/574bc6a9-a9d8-4394-b648-397f6e5e599f)（中信一级行业动量区分度高）；[行业动量轮动研报复现](https://zhuanlan.zhihu.com/p/29986910934) |
| D2 | `ind_mom_rank_60` 行业动量排名 | 行业 `r_60` 在全部行业中的分位 | 中长期行业动量（6~12M）在 A 股显著 | [复旦硕士论文:A股行业动量策略研究](https://cdmd.cnki.com.cn/Article/CDMD-10246-1013102179.htm)；基金研报共识"行业 1M 动量、6M 反转"见[华安基金研报](https://www.huaan.com.cn/upload2010/2020/11/03/d033d213-639d-3b5c-8462-7381b183a12f.pdf) |
| D3 | `ind_breadth_ma20` 行业内宽度 | 行业内 `C > MA20` 的成分股占比 | 行业整体止跌企稳时个股背离更可信 | 宽度逻辑的行业内版本，来源同 C6-C8 |
| D4 | `ind_ret_5_reversal` 行业短期反转状态 | `r_5(ind)` 的横截面分位 | 行业短期超跌（分位<0.1）+ 行业长期动量强 = 最佳背离语境 | [汉斯出版社:基于A股价量因子的行业轮动](https://www.hanspub.org/journal/paperinformation?paperid=74516)（30 日短期反转年化 8.77%）；[动量与反转效应来源分解](https://doc.taixueshu.com/journal/20140387sclyj.html) |

## 五、反转/超跌反弹文献经典语境（9 个）

| # | 特征名 | 精确公式 | 方向假设 | 证据/来源 |
|---|--------|----------|----------|-----------|
| E1 | `rsi_6` / `rsi_14` 超卖程度 | Wilder RSI，`rsi_6[T]` 与 `rsi_14[T]` 并列 | 短周期 RSI 深度超卖（<20）后反弹期望显著为正；长周期超卖确认趋势级超卖 | Connors RSI(2) 均值回归体系：[QuantifiedStrategies: RSI-2](https://www.quantifiedstrategies.com/rsi-2-strategy/)；[PapersWithBacktest: Connors RSI](https://paperswithbacktest.com/course/connors-rsi)；[掘金:BIAS+WR双指标超跌策略](https://juejin.cn/post/7664150646268346422) |
| E2 | `rsi_divergence` RSI 背离哑变量 | `L[i2] < L[i1]` 且 `RSI14[i2] > RSI14[i1]` | MACD+RSI 双背离是最可靠的背离确认组合 | 学术侧：RSI+MACD 交叉确认最可靠，[J2T: Divergence in Technical Analysis](https://j2t.com/solutions/blogview/divergance/)（无过滤假信号 30-40%）；[arXiv:2206.12282](https://arxiv.org/pdf/2206.12282/1000)（MACD+RSI/MFI 组合胜率显著提升） |
| E3 | `bias_20` 20 日乖离率 | `(C[T] - MA20[T]) / MA20[T]` | 深度负乖离（<-10%）=短期超跌极值，均值回归期望强 | [雪球:乖离率量化股价回归动能](https://xueqiu.com/6114319827/346991715)；[BIAS超跌反弹批量回测 5534 标的](https://www.zpyztech.com/bias-oversold-rebound-batch-backtest-20260731/) |
| E4 | `bias_60` 60 日乖离率 | `(C[T] - MA60[T]) / MA60[T]` | 中期乖离深度决定反弹空间上限 | [信易科技:均值回归策略](https://www.shinnytech.com/articles/trading-strategy/mean-reversion/mean-reversion-strategy) |
| E5 | `lower_shadow_l2` 次低点下影线 | `(min(O[i2],C[i2]) - L[i2]) / ATR14[i2]`；T 日版本 `lower_shadow_T` 同理 | 低点日长下影=盘中承接实锤；与锤子线统计一致 | Bulkowski：锤子线作为牛市反转成功率约 60%，下影/实体≥2 且靠近年内低点时最佳，[Bulkowski on the Hammer](https://thepatternsite.com/Hammer.html)；确认后 60-68%，[Journalplus: Hammer](https://journalplus.co/patterns/hammer)；反面证据：[NIFTY 50 锤子线研究](https://ijsred.com/volume8/issue3/IJSRED-V8I3P275.pdf)、巴西 walk-forward 仅 37-42% 胜率（见[巴西硕士论文](http://tede2.uefs.br:8080/bitstream/tede/683/2/DissertaçãoMestradoVersãoFinalCD.pdf)）——单独用弱、作语境变量仍有信息 |
| E6 | `hammer_l2` 锤子线哑变量 | `下影 ≥ 2×|实体|` 且 `上影 ≤ 0.3×(H-L)`，于 `i2` 或 `T` 日判定 | 标准锤子线定义，配合背离语境期望优于 Bulkowski 单测基线 | 同 E5 |
| E7 | `down_day_streak` 连阴/阴线密度 | T 前最长连续 `C<O` 天数；及过去 10 日阴线占比 | 连阴末端的背离=恐慌释放充分；但连阴 >7 天需警惕趋势性崩坏（非单调） | [叩富网:量价背离分析](https://licai.cofool.com/ask/qa_6922420_1_3.html)类实战文章中"超跌形态"的量化版本 |
| E8 | `kdj_j` KDJ-J 值 | `RSV=(C-LLV_9(L))/(HHV_9(H)-LLV_9(L))×100`；`K=EMA_3(RSV)`（SMA 递推）、`D=EMA_3(K)`、`J=3K-2D` | J<0 为极端超卖，国内社区最常用超卖度量之一，与 RSI 相关性中等可互补 | [财金股:量化交易指标](https://m.caijingu.com/news/news-hxttocyi.html)；中文公式社区标配 |
| E9 | `vol_price_div` 量价背离度 | 下跌段（i1→i2）内：`corr(pct_chg, V 变化率)` 或简化版 `MA_5(V)[i2]/MA_5(V)[i1]`（<1 为跌时缩量） | 二次探底量能显著小于首次=抛压衰竭的直接证据 | [百度百科:量价背离](https://baike.baidu.com/item/%E9%87%8F%E4%BB%B7%E8%83%8C%E7%A6%BB/4866454)；[叩富网:股票量价背离怎么看](https://licai.cofool.com/ask/qa_7448683.html) |

## 汇总与硬约束确认

- 全部 50 个特征均只使用 `t ≤ T` 的日线 OHLCV、MACD 衍生量及 T 日横截面数据，**T 日收盘后可算**。
- 行业组（D1-D4）需先补充行业分类数据；市场组指数特征需补下载 000300.SH（或用已有的 000001.SH/399001.SZ 替代）。
- 宽度特征（C6-C9）用 universe 内 5891 只股票自算，无外部依赖。
- 方向性标注均为假设，最终以标签命中率的实证为准；已知负证据：单一 K 线形态、单一 MACD 信号无边缘（[arXiv:2206.12282](https://arxiv.org/pdf/2206.12282/1000) 中 MACD 单独胜率 <50%；巴西锤子线样本外 37-42%），这些特征的价值在于与背离事件的**交互**而非单独使用。

## 我判断最有区分度的 15 个（按优先级）

1. **A9 `div_hist_area_shrink`** 柱线负面积收缩——背离的"量能级"量化，比单纯 DIF 抬升更本质。
2. **A8 `div_count_120`** 第几次背离——中文统计与实战共识最强的单变量。
3. **A12 `rebound_from_l2`** 初始反弹力度——区分"有承接的背离"与"死猫"的最直接变量。
4. **A11 `div_golden_cross_state`** 金叉确认状态——背离是否已被价格行为确认。
5. **A3 `div_price_newlow_depth`** 新低幅度——浅新低 vs 恐慌新低的核心区分。
6. **B12 `vol_dryup_5_60`** 量能枯竭比——基线已证有效，连续化后信息更多。
7. **E2 `rsi_divergence`** RSI 双背离——学术文献中最稳的背离增强器。
8. **C8 `breadth_pct_above_ma20`** 站上 20 日线占比——市场冰点/修复的状态变量。
9. **C1 `idx_trend_ma60`** 大盘趋势——背离胜率的市场状态开关。
10. **B4 `trend_slope_120`** 个股长期趋势斜率——基线"仅上涨段"的连续化。
11. **E3 `bias_20`** 20 日乖离率——超跌幅度与反弹空间的直接度量。
12. **B7 `vol_contraction_60`** 波动收缩比——抛压衰竭 vs 下跌加速的分水岭。
13. **B1 `pos_52w`** 52 周区间位置——锚定效应与"接飞刀 vs 黄金坑"的坐标。
14. **D1 `ind_rel_strength_20`** 行业相对强度——A 股行业动量显著，顺风/逆风区分（依赖补数据）。
15. **A1 `div_span_bars`** 背离跨度——结构噪音与有效结构的分界。
