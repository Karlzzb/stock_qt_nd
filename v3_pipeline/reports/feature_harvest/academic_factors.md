# 学术因子收割：A 股短期（5-30 交易日）截面收益预测特征全集

> 用途：MACD 底背离事件池（信号日 T，T+1 开盘买入，标签 = 20 日内涨幅触及 +2×ATR(14)）的二次排序候选特征库。
> 硬约束：所有特征必须"信号日 T 收盘后可算"，即只使用 T 及之前的行情数据。
> 检索时间：2026-09。文献范围以 2015-2026 为主，经典原始文献（Amihud 2002、Bali 2011 等）作为方法源头一并收录。

## 通用记号约定

- `C_t, O_t, H_t, L_t, V_t, Amt_t`：日收盘价、开盘价、最高价、最低价、成交量、成交额。
- `r_t = C_t / C_{t-1} - 1`：日收益率。
- `r_on,t = O_t / C_{t-1} - 1`：隔夜收益；`r_id,t = C_t / O_t - 1`：日内收益。
- `turn_t = V_t / 流通股本_t`：换手率。
- `N`：回看窗口，默认 20 个交易日（约 1 个月），与标签窗口对齐。
- 所有求和/均值默认在 `t ∈ [T-N+1, T]` 区间计算。
- **T 收盘可算**："是" = 日线 OHLCV/成交额即可；"是(分钟)" = 需分钟级数据；"是(分单)" = 需逐笔/分单数据。

---

## 家族 1：动量 / 反转（8 个）

A 股文献的一致结论：**短期反转强、中期动量弱或不存在**（散户结构 + T+1 + 卖空约束所致）。
这是与本任务（超跌反弹狙击）最契合的家族。

### 1.1 RET20 —— 20 日短期反转
- 公式：`RET20 = C_T / C_{T-20} - 1`，作为特征取原值，预期方向为负（跌得多 → 未来收益高）。
- 数据：日线收盘价。T 收盘可算：是。
- 来源：Jegadeesh (1990) 原始反转文献；A 股证据见 [Does short-term momentum exist in China? (Pacific-Basin Finance Journal, 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0927538X22002153)。
- 证据：A 股等权多空约 1.72%/月（t=3.64，2014-2026），经 CH-3 调整后 alpha 1.46%/月，见 [Short-Term Reversal in Chinese A-Shares](https://synapsesocial.com/papers/6a22688f763171746d54725b)；中文实证见 [个股反转策略与行业动量策略](https://www.sinoss.net/uploadfile/2017/0410/20170410110505797.pdf)。
- 失效风险：收益集中于低流动性小盘股，扣费后弱化；高换手区间反转最强，牛市中减弱。

### 1.2 RET5 —— 5 日超短期反转
- 公式：`RET5 = C_T / C_{T-5} - 1`。
- 数据：日线收盘价。T 收盘可算：是。
- 来源：同上；A 股周内反转在周频调仓下依然显著（[国金证券周频量价因子实证](https://bigquant.com/wiki/doc/yr2FzKlmOH)）。
- 证据：5 日反转 IC 高于 20 日但衰减快，适合 5-10 日持有期。
- 失效风险：与涨跌停制度交互——跌停后次日惯性下跌（价格发现延迟），并非纯反弹信号。

### 1.3 RET12_1 —— 中期动量（剔除最近 1 月）
- 公式：`RET12_1 = C_{T-20} / C_{T-240} - 1`。
- 数据：日线收盘价（需 1 年历史）。T 收盘可算：是。
- 来源：Jegadeesh & Titman (1993)；A 股证据见 [The evolvement of momentum effects in China (RIBAF, 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0275531922002197) 与 [Dissecting Momentum in China (Xu)](http://yuchenxu.com/paper/Momentum.pdf)。
- 证据：A 股 2008 年后动量基本消失、反转主导；作为控制变量/弱负向特征使用，不宜作为主信号。
- 失效风险：方向不稳定，regime 依赖强。

### 1.4 IDEAL_REV —— 理想反转（大单切割反转）
- 公式：过去 20 日中，按每日平均单笔成交金额 `每笔_t = Amt_t / 成交笔数_t` 排序；单笔金额最高的 10 日收益加总为 `M_high`，最低的 10 日为 `M_low`；`IDEAL_REV = M_high - M_low`（取原值，预期负向）。
- 数据：日线成交额 + 成交笔数。T 收盘可算：是。
- 来源：[开源证券"理想反转"因子（魏建榕团队）](https://bigquant.com/wiki/doc/krTwRtm1Qi)。
- 证据：全历史 IC -0.051、RankIC -0.060、IR≈2.5、月胜率约 78%，显著优于朴素 Ret20；逻辑为"反转之力的微观来源是大单"。
- 失效风险：研报因子公开后拥挤度上升；分笔数据口径（不同数据商）影响排序。

### 1.5 REV_DECOMP —— 隔夜/日内收益分解反转
- 公式：`CumON = Σ ln(1 + r_on,t)`，`CumID = Σ ln(1 + r_id,t)`（N=20），两个特征分开输入。
- 数据：日线开/收盘价。T 收盘可算：是。
- 来源：[A Tug of War: Overnight Versus Intraday Expected Returns (Lou, Polk, Skouras, JFE 2019)](https://www.fmg.ac.uk/sites/default/files/publications/DP744.pdf)；中国证据见 [The cross-section of intraday and overnight returns (JFE, 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0304405X21000854)。
- 证据：中国市场隔夜收益与日内收益长期符号相反、各自具有持续性；T+1 制度强化该结构；日内累计大跌 + 隔夜坚挺是较强的反弹候选形态。
- 失效风险：极端跳空事件（利好/利空公告）会污染分解；需配合剔除涨跌停日。

### 1.6 REL_REV —— 行业中性反转（相对反转）
- 公式：`REL_REV = RET20_stock - RET20_industry`（个股 20 日收益减去所属行业指数同期收益）。
- 数据：日线收盘价 + 行业分类。T 收盘可算：是。
- 来源：行业动量/个股反转分离框架，见 [个股反转策略与行业动量策略](https://www.sinoss.net/uploadfile/2017/0410/20170410110505797.pdf)。
- 证据：A 股个股反转与行业动量可分离，行业内反转更纯；剔除行业 beta 后 IC 稳定性提升。
- 失效风险：行业分类口径（申万/中信）变更；板块整体杀跌时行业中性化反而选出基本面恶化股。

### 1.7 INFO_DISC —— 信息离散度（Frog in the Pan）
- 公式：`ID = sign(RET20) × (%neg_days - %pos_days)`，其中 %neg/%pos 为 20 日内下跌/上涨天数占比。
- 数据：日线收盘价。T 收盘可算：是。
- 来源：Da, Gurun & Warachka (RFS 2014)；A 股复现见 [Replicating Anomalies in China (CAFR)](https://www.cafr-sif.com/2019/2019selected/Replicating%20Anomalies%20in%20China.pdf)。
- 证据：信息以连续小幅方式到达时反应不足更强；对底背离场景，"阴跌型"（ID 高）与"暴跌型"（ID 低）的后续路径显著不同，可直接区分反弹概率。
- 失效风险：与波动率因子相关度高，需中性化后使用。

### 1.8 HIGH52 —— 52 周高点距离
- 公式：`HIGH52 = C_T / max(H_t, t∈[T-240,T])`。
- 数据：日线最高价。T 收盘可算：是。
- 来源：George & Hwang (JF 2004)；A 股证据为**负面/弱**：[Zhang et al. 2019 与 Hou et al. 2023 指出 52 周高点效应在中国不成立](https://www.sciencedirect.com/science/article/abs/pii/S037842662400270X)。
- 证据：美股强、A 股弱；建议仅作位置控制变量（超跌深度度量），不作方向信号。
- 失效风险：A 股锚定效应被有限注意/散户行为淹没。

---

## 家族 2：波动率（7 个）

### 2.1 VOL20 —— 20 日总波动率
- 公式：`VOL20 = std(r_t, N=20)`。
- 数据：日线收盘价。T 收盘可算：是。
- 来源：Ang et al. (JF 2006) 低波动异象；A 股证据见 [Calm Stocks, Wild Hopes: 低波动异象与彩票偏好（NYU Shanghai 2025）](https://cdn.shanghai.nyu.edu/sites/default/files/honorsthesis2025_zijin_su.pdf)。
- 证据：A 股低波组合长期跑赢高波组合；高波动主要由彩票偏好与散户交易驱动。
- 失效风险：本任务标签以 ATR 为单位，高波股天然更容易触线——存在"标签-波动率机械相关"，必须作为控制变量纳入而非剔除。

### 2.2 IVOL —— 特质波动率
- 公式：对过去 60 日做 `r_i - r_f = α + β·MKT`（或 CH-3 三因子）回归，`IVOL = std(残差)`。
- 数据：日线收盘价 + 指数收益。T 收盘可算：是。
- 来源：Ang et al. (2006)；A 股证据见 [模糊是低特质波动率异象的成因吗？——来自中国股市的证据](https://html.rhhz.net/HNLGDXXBSKB/html/2020-5-53.htm)。
- 证据：A 股 IVOL 与次月收益负相关，年化多空 7.6-9.2%（见家族 7 引文）；散户占比高的股票异象更强。
- 失效风险：因子模型选择敏感（CAPM/FF3/CH-3 残差差异大）。

### 2.3 PARK —— Parkinson 高低价波动率
- 公式：`PARK = sqrt( (1/(4·ln2·N)) · Σ (ln(H_t/L_t))² )`。
- 数据：日线高/低价。T 收盘可算：是。
- 来源：Parkinson (1980)；A 股金工实证广泛用于振幅类因子构建（见 2.4）。
- 证据：比收盘-收盘波动率信息效率高约 5 倍（理论上）；在 A 股与换手、振幅因子高度互补。
- 失效风险：涨跌停日 H=L 或 H=涨停价时失真。

### 2.4 AMP —— 平均振幅因子
- 公式：`AMP = mean( (H_t - L_t) / C_{t-1}, N=20 )`，预期负向（高振幅 → 未来收益低）。
- 数据：日线高/低/收盘价。T 收盘可算：是。
- 来源：[开源证券"理想振幅"体系](http://mp.weixin.qq.com/s?__biz=MzI1NTYxMjE1Mw==&mid=2247602968&idx=1&sn=ddcf3ff69f39bdf8ff32ed6580e361ab)。
- 证据：理想振幅因子 IC -0.054、RankIC -0.072、IR≈3.0、月胜率约 84%，是开源四因子中最强；分钟切割版本 rankICIR 达 -4.58。
- 失效风险：与 VOL20 相关度高，建议二者取其一或正交化。

### 2.5 IDEAL_AMP —— 理想振幅（价态切割）
- 公式：20 日内按收盘价所处价态（高/低价态，如 `C_t / max(C,N)` 分位）分两组，高价位日振幅均值 `V_high`，低价位日 `V_low`，`IDEAL_AMP = V_high - V_low`。
- 数据：日线 OHLC。T 收盘可算：是。
- 来源：同上（开源金工）。
- 证据：高价态下的高振幅（高开低走/放量滞涨）含最强负面信息；对"超跌后是否真反弹"有直接判别力。
- 失效风险：价态定义（切割比例 λ）对结果敏感，需做参数稳健性检验。

### 2.6 DVOL —— 波动率变化
- 公式：`DVOL = std(r, 5) - std(r, 60)`（短窗波动减长窗波动）。
- 数据：日线收盘价。T 收盘可算：是。
- 来源：波动率突变作为信息到达代理；A 股换手率变动类研究的对偶，见 [流动性变动与股票横截面收益的关系](https://www.zhangqiaokeyan.com/academic-degree-domestic_mphd_thesis/020313780023.html)。
- 证据：波动率骤升伴随分歧加大、后续收益偏低；波动率收缩后的放量突破是经典有效形态。
- 失效风险：与事件信号（底背离本身即波动形态）共线。

### 2.7 ATRN —— 归一化 ATR 水平
- 公式：`ATRN = ATR(14)_T / C_T`，其中 ATR(14) 为 True Range 的 14 日均值。
- 数据：日线 OHLC。T 收盘可算：是。
- 来源：Wilder (1978) 原始定义；作为标签阈值单位，必须显式建模。
- 证据：非预测因子，但标签 `+2×ATR` 的可达性直接由 ATRN 决定；高 ATRN 股票触线所需绝对涨幅更大。
- 失效风险：无（这是必须的交互/分层变量，不是 alpha 因子）。

---

## 家族 3：流动性 / 换手（8 个）

### 3.1 TURN20 —— 20 日平均换手率
- 公式：`TURN20 = mean(turn_t, N=20)`，预期负向（低换手 → 高收益）。
- 数据：日线成交量 + 流通股本。T 收盘可算：是。
- 来源：[张峥、刘力 (2006)《换手率与股票收益：流动性溢价还是投机性泡沫？》](https://ccj.pku.edu.cn/article/info?id=301451506)；[Liu, Stambaugh & Wei (2019) CH-3 九类异象](http://mp.weixin.qq.com/s?__biz=MzU3MDY2ODU3Mg==&mid=2247500582)。
- 证据：A 股最强异象之一；换手最高组与最低组收益差 -2.46%/月（t=-4.39，见 [Does short-term momentum exist in China?](https://www.sciencedirect.com/science/article/abs/pii/S0927538X22002153)）；机制为异质信念 + 卖空约束下的高估。
- 失效风险：与规模高度共线；[换手率呈不对称倒 U 形](https://qks.sufe.edu.cn/mv_html/j00003/201805/136942e3-4ab3-4dff-a60f-99e23aec1b58_WEB.htm)，极低换手段方向反转，建议用分位秩而非线性值。

### 3.2 ABTURN —— 异常换手率（换手率变化）
- 公式：`ABTURN = mean(turn, 5) / mean(turn, 60)` 或 `turn_T - mean(turn, 120)`。
- 数据：同上。T 收盘可算：是。
- 来源：[中国A股市场流动性冲击与股票回报率关系研究（TURNS 定义）](https://www.sciengine.com/doi/pdf/D54CF624146F4DF28777CAFEBD4986E6)。
- 证据：换手率变动对下月收益有独立预测力，正向变动预测力强于负向；底背离场景下"缩量见底 → 温和放量"是核心确认形态。
- 失效风险：放量既可能是反弹启动也可能是出货，需与量价相关因子联用。

### 3.3 STDTURN —— 换手率波动率
- 公式：`STDTURN = std(turn_t, N=20)`。
- 数据：同上。T 收盘可算：是。
- 来源：[中国A股市场异象性研究（清华五道口，2018，56 异象检验）](https://www.pbcsf.tsinghua.edu.cn/info/1510/8491.htm)；[A股市场的时变多因子模型](https://xbgjxt.swu.edu.cn/data/article/preview-pdf?doi=10.13718/j.cnki.xdsk.2022.06.012)。
- 证据：换手波动因子在时变 LASSO 中 32 次入选（换手水平因子仅 4 次），稳健性远超换手率水平；预期负向。
- 失效风险：与波动率因子簇共线，建议正交化。

### 3.4 ILLIQ —— Amihud 非流动性
- 公式：`ILLIQ = mean( |r_t| / Amt_t, N=20 )`（通常取对数）。
- 数据：日线收益 + 成交额。T 收盘可算：是。
- 来源：Amihud (2002)；A 股证据见 [Are conditional illiquidity risks priced in China? (IREF 2022)](https://www.sciencedirect.com/science/article/abs/pii/S1057521922000497) 与 [Aggregate liquidity premium and cross-sectional returns: Evidence from China (Economic Modelling 2021)](https://www.sciencedirect.com/science/article/pii/S0264999321002340)。
- 证据：A 股存在稳健正的非流动性溢价；溢价集中在高 ILLIQ、低价、高方差比股票，被解释为行为误定价而非纯风险补偿。
- 失效风险：与规模因子共线极强；涨跌停制度下 |r| 被截断会低估真实非流动性。

### 3.5 AMT —— 成交额规模
- 公式：`AMT = ln( mean(Amt_t, N=20) )`。
- 数据：日线成交额。T 收盘可算：是。
- 来源：[清华五道口 56 异象检验](https://www.pbcsf.tsinghua.edu.cn/info/1510/8491.htm)（交易额为 11 个有效因子之一）。
- 证据：低成交额股票溢价显著，属交易摩擦类因子；同时是容量/可交易性硬指标。
- 失效风险：主要为规模代理，alpha 含量低；作为过滤条件（可交易性）优于作为排序特征。

### 3.6 VR —— 量比
- 公式：`VR = mean(V_t, 5) / mean(V_t, 60)`。
- 数据：日线成交量。T 收盘可算：是。
- 来源：A 股技术分析实证（见 [技术指标在中国有效性研究（eScholarship）](https://escholarship.org/content/qt1p90t92h/qt1p90t92h.pdf)）；与 ABTURN 同源但用成交量口径。
- 证据：作为短期量能异动度量广泛有效；方向依赖价格位置（底部放量正向、顶部放量负向）。
- 失效风险：单独使用方向不稳定，必须与收益符号交互。

### 3.7 FLOAT_MCAP —— 自由流通市值
- 公式：`FLOAT_MCAP = ln(C_T × 流通股本_T)`。
- 数据：收盘价 + 流通股本。T 收盘可算：是。
- 来源：[Cakici et al. (2017) Cross-sectional stock return predictability in China](https://scholars.cityu.edu.hk/en/publications/cross-sectional-stock-return-predictability-in-china/)（规模为中国最强预测变量之一）。
- 证据：A 股规模效应全球罕见地强（壳价值 + 散户小票偏好）；小市值股票反转/反弹弹性更大。
- 失效风险：2017 后、2021 后规模效应多次阶段性反转；注册制扩容持续压缩壳价值。

### 3.8 PRC —— 股价水平
- 公式：`PRC = ln(C_T)`。
- 数据：收盘价。T 收盘可算：是。
- 来源：同上（Cakici et al. 2017，price 为显著预测变量）；低价股与彩票偏好相关（见家族 7）。
- 证据：低价股在 A 股具有彩票属性与更高投机弹性；同时是退市风险代理（面值退市规则 1 元）。
- 失效风险：非单调，极低价段混入退市风险标的。

---

## 家族 4：量价相关（4 个）

### 4.1 CPV —— 日频量价相关系数
- 公式：`CPV = corr(C_t, V_t, N=20)`（或 corr(收盘价, 成交额)），预期负向（相关性低/背离 → 未来收益高）。
- 数据：日线收盘价 + 成交量。T 收盘可算：是。
- 来源：[东吴证券《高频价量相关性，意想不到的选股因子》(2020)](https://asset.quant-wiki.com/pdf/20200223-东吴证券-“技术分析拥抱选股因子”系列研究（一）：高频价量相关性，意想不到的选股因子.pdf)；[海通证券选股因子系列12《"量"与"价"的结合》](https://asset.quant-wiki.com/pdf/海通选股因子系列研究12：“量”与“价”的结合.pdf)。
- 证据：高频版 CPV 月 IC -0.053、年化 ICIR -3.77、多空年化 19.29%、胜率 87%；日频简化版方向一致、强度打折。
- 失效风险：本质为反转因子变体，与 RET20 相关；空头端贡献约 70%，对纯多头筛选的增益需谨慎评估。

### 4.2 RVC —— 收益-量变相关
- 公式：`RVC = corr(r_t, V_t, N=20)`，预期负向。
- 数据：日线收益 + 成交量。T 收盘可算：是。
- 来源：同上（东吴/海通系列）。
- 证据：与 CPV 互补；`corr(r,V)<0`（下跌放量/上涨缩量）在底部区域反而是恐慌出清信号，方向需结合位置。
- 失效风险：在涨跌停制度下量被压制，相关系数估计有偏。

### 4.3 CPV_MIN —— 分钟级量价相关（均值/波动/趋势三维）
- 公式：每日计算分钟收盘价与分钟成交量相关系数，再对 20 日的日度序列取均值、标准差、斜率，三特征。
- 数据：分钟线。T 收盘可算：是(分钟)。
- 来源：东吴证券 2020（上引）；[国金证券《基于高频快照数据的量价背离选股因子》(2022)](https://bigdata-s3.wmcloud.com/researchreport/2022-11/713e883d673d20257c1eb235cf8f657d.pdf)。
- 证据：国金快照版合成因子中性化后周频 IC 6.27%、多空夏普 4.08，已用于中证1000指数增强；为 A 股近年最强量价因子族。
- 失效风险：数据成本高；高频因子衰减快，研报公开后拥挤。

### 4.4 CPV_TREND —— 量价相关的趋势
- 公式：`CPV_TREND = slope(corr(C,V,10)_t, N=10)`（滚动量价相关系数的线性趋势）。
- 数据：日线。T 收盘可算：是。
- 来源：东吴证券三维 CPV 中的趋势维度；[价量相关性CPV因子绩效月报](https://bigquant.com/wiki/doc/TeGyYHKuqe)。
- 证据：相关性由正转负（量价走向背离）本身是情绪拐点信号，与 MACD 底背离事件天然互补。
- 失效风险：估计噪声大，窗口敏感。

---

## 家族 5：量价背离类（3 个）

> 注意：事件本身（MACD 底背离）即属此家族；本节特征用于刻画背离的"质量/形态"，与事件形成正交信息。

### 5.1 OBV_DIV —— OBV 与价格的背离度
- 公式：`OBV_T = OBV_{T-1} + sign(r_T)·V_T`；背离度 = `zscore(C,20) - zscore(OBV,20)`（价格相对位置减去 OBV 相对位置）。
- 数据：日线收盘价 + 成交量。T 收盘可算：是。
- 来源：Granville OBV 原始定义；背离类金工实证见 [量价背离因子（BigQuant/国金方法综述）](https://bigquant.com/wiki/doc/Hn333yYkfS)。
- 证据：价创新低而 OBV 未创新低（资金未同步流出）是经典的底部确认；金工分组实证中因子值最小组（深度背离）收益最高。
- 失效风险：OBV 对单日巨量敏感（一字板日扭曲累积值）。

### 5.2 PVR —— 正负日量比
- 公式：`PVR = mean(V_t | r_t>0) / mean(V_t | r_t<0)`（20 日内上涨日均量 / 下跌日均量）。
- 数据：日线。T 收盘可算：是。
- 来源：量价背离细分因子族（[量价背离细分因子收益表现跟踪](https://www.hangyan.co/charts/3502044978702451852)）。
- 证据：底部区域 PVR 上升（反弹日放量、阴跌日缩量）是吸筹形态；顶部相反。
- 失效风险：20 日内上涨日数过少时估计不稳定（需加最小样本约束）。

### 5.3 VSHRINK —— 下跌段量能衰减度
- 公式：`VSHRINK = mean(V_t | r_t<0, 近10日) / mean(V_t | r_t<0, 前10日)`（近期下跌日量 / 早期下跌日量），值小表示缩量阴跌。
- 数据：日线。T 收盘可算：是。
- 来源：背离/出清逻辑，见 [量价背离因子：提前识别市场情绪变化](https://zhuanlan.zhihu.com/p/6223195931) 及开源交易行为因子体系。
- 证据：缩量探底是卖压衰竭的直接度量，与底背离信号的条件概率高度相关。
- 失效风险：无量阴跌也可能是流动性枯竭（无人问津），需与 ILLIQ 联合判断。

---

## 家族 6：波动率结构（4 个）

### 6.1 DSR —— 下行波动占比
- 公式：`DSR = Σ min(r_t,0)² / Σ r_t²`（N=20），日频近似的半方差比。
- 数据：日线收盘价。T 收盘可算：是。
- 来源：Bollerslev, Li & Zhao (2020) good/bad volatility；A 股高频实证见 [高频因子综述（下行波动占比 ICIR 3.31）](https://blog.csdn.net/zhangyunchou2015/article/details/147247860)。
- 证据：下行波动占比与未来收益正相关（A 股高频版 ICIR 约 3.3），与直觉相反但实证稳健——下跌波动大者补偿更高。
- 失效风险：日频近似与分钟版差距大；与 IVOL 共线。

### 6.2 SJV —— 符号跳跃变差
- 公式：`SJV = Σ (r_t² · 1{r_t>θ}) - Σ (r_t² · 1{r_t<-θ})`，`θ = 2×std(r,60)`（日频近似的符号跳跃）。
- 数据：日线收盘价。T 收盘可算：是。
- 来源：[Sign realized jump risk and the cross-section of stock returns: Evidence from China (PLOS ONE 2017)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0181990)。
- 证据：中国市场跳跃比发达市场更大更频繁（约 40% 交易日含跳跃，见 [ jumps in realized volatility, PMC 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7147854/)）；负跳跃风险在截面获正补偿。
- 失效风险：日频只能识别大跳跃；阈值 θ 敏感。

### 6.3 JUMPFREQ —— 跳跃频率
- 公式：`JUMPFREQ = mean( 1{|r_t| > 2×std(r,60)}, N=60 )`。
- 数据：日线收盘价。T 收盘可算：是。
- 来源：同上（中国跳跃文献）。
- 证据：高跳跃频率 = 高信息不确定性/高投机性，与彩票因子簇相关，预期负向。
- 失效风险：与 MAX 因子高度相关。

### 6.4 RSV —— 高频半方差分解（RS+ / RS-）
- 公式：分钟收益平方按符号分别加总得 `RS+`、`RS-`，20 日均值及比值 `RS-/RS+`。
- 数据：分钟线。T 收盘可算：是(分钟)。
- 来源：[A new measure of realized volatility: inertial and reverse realized semivariance (FRL 2022, 沪深300 实证)](https://www.sciencedirect.com/science/article/abs/pii/S1544612321005882)；[Time series momentum and reversal: intraday information from realized semivariance (PBFJ 2023)](https://www.sciencedirect.com/science/article/abs/pii/S0927539823000245)。
- 证据：反向半方差（RRV）与反向跳跃变差（RJV）有显著收益预测力；中国数据上负半方差信息含量高于正半方差。
- 失效风险：分钟数据质量（集合竞价、午休）处理不当会引入噪声。

---

## 家族 7：偏度 / 高阶矩 / 彩票（6 个）

### 7.1 MAX —— 20 日最大日收益
- 公式：`MAX = max(r_t, N=20)`，预期负向。
- 数据：日线收盘价。T 收盘可算：是。
- 来源：Bali, Cakici & Whitelaw (JFE 2011)；A 股证据见 [The MAX Effect in China's A-Share Market (SCIRP)](https://www.scirp.org/journal/paperinformation?paperid=126925) 与 [Research on Investment Preference and the MAX Effect in Chinese Stock Market](https://www.degruyter.com/document/doi/10.21078/JSSI-2016-519-15/pdf)。
- 证据：A 股 MAX 效应强：高 MAX 股票未来收益显著低，高 IVOL、正偏、彩票属性集中；投资者情绪高涨期效应放大。
- 失效风险：与"近 20 日涨停次数"（家族 12）部分重叠；极端市况下 MAX 全是涨停噪声。

### 7.2 MAX5 —— 最大 5 日收益均值
- 公式：`MAX5 = mean(最大的5个 r_t, N=20)`。
- 数据：日线。T 收盘可算：是。
- 来源：Bali et al. (2011) 稳健性版本；A 股金工常规采用。
- 证据：比单日 MAX 平滑，IC 略低但更稳。
- 失效风险：同 7.1。

### 7.3 MIN —— 20 日最小日收益
- 公式：`MIN = min(r_t, N=20)`，预期正向（跌得狠 → 反弹）。
- 数据：日线。T 收盘可算：是。
- 来源：Bali et al. (2011) 的 MIN 对应物；A 股验证见 [Replicating Anomalies in China](https://www.cafr-sif.com/2019/2019selected/Replicating%20Anomalies%20in%20China.pdf)。
- 证据：与 MAX 不对称：极端下跌后的反转补偿在 A 股显著；与本任务"超跌反弹"标签方向一致。
- 失效风险：跌停连板股票 MIN 极低但不可买（T+1 无法成交），需配合可交易性过滤。

### 7.4 SKEW60 —— 收益偏度
- 公式：`SKEW60 = skew(r_t, N=60)`。
- 数据：日线。T 收盘可算：是。
- 来源：Zhang (2005) 偏度偏好；A 股证据见 [Skewness Preference and IPO Anomalies in China](http://aeconf.com/Articles/May2017/aef180108.pdf)。
- 证据：A 股投资者偏好正偏股票导致其被高估、未来收益低；MAX 常被视为偏度的廉价代理。
- 失效风险：60 日偏度估计噪声大；A 股偏度异象弱于 MAX/IVOL（[Replicating Anomalies in China](https://www.cafr-sif.com/2019/2019selected/Replicating%20Anomalies%20in%20China.pdf) 结论）。

### 7.5 KURT60 —— 收益峰度
- 公式：`KURT60 = kurtosis(r_t, N=60)`。
- 数据：日线。T 收盘可算：是。
- 来源：[Forecasting stock market volatility: Do realized skewness and kurtosis help? (Physica A)](https://www.sciencedirect.com/science/article/abs/pii/S0378437117303205)（波动预测维度）；截面维度证据较弱。
- 证据：高峰度 = 厚尾/跳跃密集，作为风险形态变量使用；截面方向证据不一致。
- 失效风险：估计噪声极大；建议仅作交互项。

### 7.6 IVOV —— 特质波动的波动（vol of vol）
- 公式：`IVOV = std( rolling_std(r,5), N=60 )` 或对日度 IVOL 序列取 std。
- 数据：日线。T 收盘可算：是。
- 来源：[The idiosyncratic volatility of volatility effect in the A-share market (FRL 2025)](https://www.sciencedirect.com/science/article/pii/S1544612325001163)。
- 证据：A 股特异"波动的波动"效应：高 IVOV 股票年化跑输 7.6-9.2%，机制为异质方差信念。
- 失效风险：窗口嵌套导致有效样本需求大；与 STDTURN 相关。

---

## 家族 8：日历效应（5 个）

### 8.1 DOW —— 星期几哑变量
- 公式：信号日 T（或买入日 T+1）的星期几 one-hot（重点：周四 = 1、周五 = 1）。
- 数据：交易日历。T 收盘可算：是。
- 来源：[沪深股市周内效应再检验（重庆大学学报 2014）](http://qks.cqu.edu.cn/html/cqdxskcn/2014/3/201403006.htm)；[融券卖空与周末效应](https://gcxb.gufe.edu.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=8815)。
- 证据：A 股"红周一、黑周四"（T+1 制度下周四卖出规避周末不确定性）；熊市中周一、周四显著偏弱；牛市周一显著偏强。
- 失效风险：周内效应 regime 依赖极强（牛熊方向相反），必须与家族 9 交互。

### 8.2 MONTHEND —— 月初/月末位置
- 公式：买入日在月内的相对位置（距月末交易日数，或月初 5 日 dummy）。
- 数据：交易日历。T 收盘可算：是。
- 来源： turn-of-month 效应国际文献（Lakonishok & Smidt 1988）；A 股实证散见券商择时研究（[海通：周内效应与市场状态择时](https://finance.sina.com.cn/stock/stockzmt/2020-03-18/doc-iimxyqwa1302257.shtml)）。
- 证据：A 股月末偏弱（资金面考核）、月初偏强（资金回流）。
- 失效风险：效应量级小，单独不可交易，只作条件变量。

### 8.3 HOLIDAY —— 节前/节后哑变量
- 公式：买入日距长假（春节/国庆）前后 N 日的 dummy（节前 10 日、节后 5 日）。
- 数据：交易日历。T 收盘可算：是。
- 来源：[The weekly cycle of investor sentiment and the holiday effect（中国情绪"5+2"周期）](https://pmc.ncbi.nlm.nih.gov/articles/PMC9816973/)；[Pre-Holiday Effect (Quantpedia)](https://quantpedia.com/strategies/pre-holiday-effect)。
- 证据：A 股国庆前 10 日上涨概率不足三成，节后 T+1 至 T+5 胜率六至八成；春节后 5 日胜率约 72%、均值 +1.8%。
- 失效风险：样本极少（每年 2 次长假），统计功效低；年份间方差大。

### 8.4 MONTH —— 月份哑变量
- 公式：月份 one-hot（重点：2 月、6 月、11-12 月）。
- 数据：交易日历。T 收盘可算：是。
- 来源：A 股"春季躁动"与"五穷六绝七翻身"实证综述（[日历效应盘点](http://mp.weixin.qq.com/s?__biz=MzUzNzU3MzM0OQ==&mid=2247487751&idx=1&sn=84aad90b616aa912ecb3f1f91c2e9cc3)）。
- 证据：2 月上证上涨概率 15 年超 75%、均值 +3.8%；6 月最弱；11-12 月机构排名行情。
- 失效风险：属市场级变量，对截面的解释力需通过"小盘/题材 × 月份"交互实现。

### 8.5 TRADEDAY_SEQ —— 月内/周内时序序号
- 公式：买入日为当月第几个交易日、当周第几个交易日（连续值）。
- 数据：交易日历。T 收盘可算：是。
- 来源：与 8.1/8.2 同族，连续化版本便于树模型使用。
- 证据：同 8.1/8.2。
- 失效风险：与哑变量版冗余，建模时二选一。

---

## 家族 9：市场状态 / Regime（6 个）

### 9.1 MKT_RET20 —— 市场 20 日收益
- 公式：`MKT_RET20 = Index_T / Index_{T-20} - 1`（沪深300/中证全指）。
- 数据：指数日线。T 收盘可算：是。
- 来源：市场状态条件化的异象研究（崔婧等 2008 牛熊周内效应；[MAX 效应随投资者情绪放大](https://www.scirp.org/journal/paperinformation?paperid=126925)）。
- 证据：A 股几乎所有短周期异象强度都随市场状态变化；反转/反弹策略在震荡市最强、单边熊市初期最差。
- 失效风险：指数选择（300/500/1000/全指）影响结论，建议多指数并行。

### 9.2 MKT_VOL —— 市场波动状态
- 公式：`MKT_VOL = std(r_index, 20)` 或其 250 日分位。
- 数据：指数日线。T 收盘可算：是。
- 来源：[International volatility risk and Chinese stock return predictability (JIMF 2017)](https://www.sciencedirect.com/science/article/abs/pii/S0261560616301085)（ΔVIX 预测中国次日收益）。
- 证据：高波动 regime 下反弹幅度大但触线失败率（继续破位）也更高；波动分位比水平更稳。
- 失效风险：波动聚集导致状态切换滞后。

### 9.3 BREADTH —— 市场广度
- 公式：`BREADTH = mean( 当日上涨家数 / 总家数, N=20 )`（全市场截面，每日聚合）。
- 数据：全市场日线涨跌。T 收盘可算：是（需全市场截面，事件研究本身已有）。
- 来源：市场宽度作为情绪/状态代理的常规用法；A 股情绪研究（[清华五道口异象研究](https://www.pbcsf.tsinghua.edu.cn/info/1510/8491.htm) 交易摩擦类因子亦以全市场聚合为条件）。
- 证据：广度极值（<20% 或 >80%）对应反弹/回落概率显著偏移；对 20 日窗口标签尤其实用。
- 失效风险：指数权重失真（大票护盘时广度与指数背离）。

### 9.4 LIMITUP_N —— 市场涨停家数（情绪温度计）
- 公式：当日全市场涨停家数、跌停家数及其比值/20 日均值。
- 数据：全市场日线 + 涨跌停价规则。T 收盘可算：是。
- 来源：A 股特有情绪代理，见 [中国A股涨跌停交易制度与投资者处置效应（金融研究）](http://www.jryj.org.cn/CN/abstract/abstract1373.shtml) 的制度背景。
- 证据：涨停家数是游资/散户情绪最直接的日频温度计；冰点（涨停<20 家）后 5 日反弹胜率显著高。
- 失效风险：2019 后注册制板块 20cm 涨停改变阈值口径，需分板块计算。

### 9.5 MKT_MA —— 指数趋势状态
- 公式：`MKT_MA = 1{Index_T > MA(Index,60)}` 及 `Index_T / MA60 - 1`。
- 数据：指数日线。T 收盘可算：是。
- 来源：均线择时的 A 股有效性（[技术指标在中国更有效（eScholarship）](https://escholarship.org/content/qt1p90t92h/qt1p90t92h.pdf)：8 个指标在中国全部比美国更准，因散户行为自我强化）。
- 证据：指数在 MA60 下方时个股反弹的持续性与胜率显著不同；熊市中的底背离失败率高。
- 失效风险：震荡市反复穿越均线，状态噪声大。

### 9.6 RETAIL_SENT —— 散户情绪代理（换手加总）
- 公式：全市场换手率加总的 20 日均值及其 250 日分位。
- 数据：全市场换手。T 收盘可算：是。
- 来源：Baker & Wurgler 情绪框架的中国版——换手率是 A 股情绪第一代理（[张峥、刘力 2006](https://ccj.pku.edu.cn/article/info?id=301451506) 的投机性泡沫解释）。
- 证据：高情绪期彩票/高换手股后续跑输更严重；低情绪冰点利好超跌反弹类策略。
- 失效风险：长期趋势（换手率中枢下移）需去趋势。

---

## 家族 10：行业 / 板块相对强度（4 个）

### 10.1 IND_MOM —— 行业 20 日动量
- 公式：`IND_MOM = IndIndex_T / IndIndex_{T-20} - 1`（所属申万/中信一级行业指数）。
- 数据：行业指数日线 + 行业分类。T 收盘可算：是。
- 来源：Moskowitz & Grinblatt (1999) 行业动量；A 股证据见 [个股反转策略与行业动量策略](https://www.sinoss.net/uploadfile/2017/0410/20170410110505797.pdf)。
- 证据：A 股行业层面动量显著强于个股层面（与美股相反的结构：个股反转、行业动量并存）。
- 失效风险：行业轮动加快（2021 后月度级轮动），20 日窗口可能偏长。

### 10.2 IND_RS_RANK —— 行业相对强度排名
- 公式：全部行业 20 日收益的横截面分位秩（0-1）。
- 数据：同上。T 收盘可算：是。
- 来源：相对强度文献（同上）；A 股板块轮动研究惯例。
- 证据：强行业中弱势股（补涨）与弱行业中超跌股（错杀修复）是两种不同 alpha 来源，排名变量便于交互建模。
- 失效风险：与 IND_MOM 冗余，二选一。

### 10.3 REL_STR —— 个股对行业超额收益
- 公式：`REL_STR = RET20_stock - RET20_industry`（与 1.6 相同公式，此处作为相对强度而非反转变量使用）。
- 数据：日线 + 行业分类。T 收盘可算：是。
- 来源：行业中性化文献惯例。
- 证据：作为"行业内相对位置"度量，方向应与 RET20 联动解释（深跌 + 行业强 = 错杀修复候选）。
- 失效风险：与 1.6 是同一变量，建模时只保留一次。

### 10.4 LINK_MOM —— 关联股动量溢出
- 公式：与该股同概念/同供应链/共同新闻提及的股票组合的 20 日平均收益。
- 数据：关联关系数据（概念板块成分可作廉价代理）。T 收盘可算：是。
- 来源：[Diamond Cuts Diamond: News Co-mention Momentum Spillover Prevails in China (JBF 2025)](https://www.sciencedirect.com/science/article/abs/pii/S037842662400270X)。
- 证据：A 股动量主要通过"关联溢出"实现（有限注意机制），新闻共提及动量显著强于传统动量；对高分析师覆盖、大流通盘、高机构持股者减弱。
- 失效风险：关联数据成本高；概念板块成分频繁调整引入前视偏差风险（需用 T 时点的历史成分）。

---

## 家族 11：资金流代理（5 个）

### 11.1 SMART —— 聪明钱因子
- 公式：分钟级：`S_t = |r_min,t| / sqrt(V_min,t)` 排序取前 20% "聪明分钟"，计算这些分钟的 VWAP 相对全天 VWAP 的偏离；`SMART = VWAP_smart / VWAP_all`。
- 数据：分钟线。T 收盘可算：是(分钟)。
- 来源：[方正证券《跟踪聪明钱：从分钟行情数据到选股因子》(2016)](https://bigquant.com/wiki/doc/V8A7T4hGwJ)；开源证券复现版 IC -0.038、RankIC -0.061、月胜率约 82%。
- 证据：机构参与度高的股票被低估后修复概率高；对事件驱动二次筛选适配性好。
- 失效风险：分钟数据成本；因子公开多年、拥挤度高。

### 11.2 MFLOW —— 主力净流入占比
- 公式：`MFLOW = Σ 大单净买入额_t / Σ Amt_t`（N=5 或 20），大单定义按数据商分单口径。
- 数据：分单资金流数据。T 收盘可算：是(分单)。
- 来源：[华泰证券《单因子测试之资金流向因子》(2018)](https://bigquant.com/wiki/doc/GaQ8lwTdtt)。
- 证据：主力净流入与未来短期收益正相关，但强度中等、衰减快（5 日窗口优于 20 日）。
- 失效风险：分单口径是算法推断而非真实席位，数据商间差异大；主力"拆单"规避导致口径失真。

### 11.3 APM —— 日内行为模式因子
- 公式：统计个股过去 20 日隔夜收益与上午收益的差异（`Δ = mean(r_on) - mean(r_am)`），标准化后取秩。
- 数据：分钟线（或开/午收/收盘价近似）。T 收盘可算：是(分钟)。
- 来源：开源证券 APM 因子（[交易行为因子体系](https://pdf.dfcfw.com/pdf/H3_AP202412021641149056_1.pdf)）：日内不同时段交易者结构不同，反转强度不同。
- 证据：IC 0.030、RankIC 0.035、月胜率约 78%；与反转/聪明钱低相关，正交代价值高。
- 失效风险：单因子强度偏弱，需组合使用。

### 11.4 TAIL_RET —— 尾盘收益占比
- 公式：`TAIL_RET = mean( r_{14:30-15:00,t} / (|r_id,t|+ε), N=20 )` 或尾盘 30 分钟收益的 20 日累计。
- 数据：分钟线。T 收盘可算：是(分钟)。
- 来源：尾盘交易含机构调仓信息（A 股尾盘 3 分钟集合竞价制度）；关联研究见 [Intraday momentum and reversal in Chinese stock market (FRL)](https://www.sciencedirect.com/science/article/abs/pii/S1544612318307414)（首半小时收益的预测力）。
- 证据：尾盘持续走强是资金抢筹信号；首半小时收益对当日剩余时段有预测力（噪声交易驱动）。
- 失效风险：尾盘 3 分钟集合竞价（2018 后深市、2019 后沪市）改变了尾盘微观结构。

### 11.5 GAP —— 隔夜跳空统计
- 公式：`GAP_MEAN = mean(r_on,t, N=20)`，`GAP_VOL = std(r_on,t, N=20)`，两特征。
- 数据：日线开/收盘。T 收盘可算：是。
- 来源：隔夜/日内分解文献（Lou, Polk & Skouras 2019；A 股 T+1 强化隔夜信息结构）。
- 证据：持续低开（GAP_MEAN<0）反映隔夜恐慌/利空消化过程，与底背离的"空头衰竭"条件相关；GAP_VOL 高 = 信息不确定性大。
- 失效风险：与 1.5 部分冗余；公告驱动跳空为事件噪声。

---

## 家族 12：A 股特有制度因子（涨跌停 / T+1 / 散户结构）（6 个）

### 12.1 LIMITCNT —— 近 20 日涨停次数
- 公式：`LIMITCNT = Σ 1{C_t ≥ 涨停价_t × 0.999}`（主板 ±10%、创业板/科创板 ±20%、ST ±5% 分规则判定）。
- 数据：日线 + 涨跌停规则。T 收盘可算：是。
- 来源：[Daily Price Limits and Destructive Market Behavior (Kim & Rhee 前身，Xiong 等)](https://www.princeton.edu/~wxiong/papers/PriceLimit.pdf)；[中国证券市场涨跌幅限制的磁力效应研究（管理科学学报）](http://jmsc.tju.edu.cn/jmsc/article/html/20080514)。
- 证据：涨停是 A 股最强的注意力/动量事件；近 20 日有涨停记录的股票投机属性强、弹性大，与 MAX 因子互补（MAX 度量幅度、LIMITCNT 度量制度性封板）。
- 失效风险：注册制后 20cm 板块与 10cm 板块行为分化，需分板计算；涨停后价格发现延迟使次日收益分布肥尾。

### 12.2 DIST_LIMIT —— 距涨停价距离
- 公式：`DIST_LIMIT = C_T / 涨停价_T`（∈(0,1]，1 = 收涨停）。
- 数据：日线 + 涨停价。T 收盘可算：是。
- 来源：磁吸效应文献（[针对高频数据的中国股市磁吸效应研究](http://clgzk.qks.cqut.edu.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=2616)）：价格越接近涨跌停，被"吸向"限制价的概率越大。
- 证据：磁吸效应在 A 股高频数据中显著；收在涨停附近但**未封板**（炸板/触板未封）与封死涨停的次日路径截然不同。
- 失效风险：磁吸证据主要来自股灾期与高频样本，常态市弱化。

### 12.3 LIMITDOWN_FLAG —— 近 20 日跌停标记
- 公式：`LIMITDOWN_FLAG = Σ 1{C_t ≤ 跌停价_t × 1.001}`。
- 数据：日线 + 跌停规则。T 收盘可算：是。
- 来源：[价格限制机制对股票价格波动及流动性的影响（北航学报）](https://bhxb.buaa.edu.cn/bhsk/cn/article/pdf/preview/8528.pdf)；[涨跌停之前的市场微观结构特征分析（股灾期研究）](https://glkx.hit.edu.cn/__local/A/50/88/B3D667C8315EBB001B9E8ABC35D_C1A7224C_4BBF1B.pdf)。
- 证据：跌停存在价格发现延迟——次日继续低开概率显著高于普通大跌日；近期跌停是底背离信号的强负向调节变量（"接飞刀"风险）。
- 失效风险：跌停原因异质（利空公告 vs 流动性踩踏），哑变量无法区分，建议与跳空/成交配合。

### 12.4 T1_GAP —— T+1 制度性隔夜行为
- 公式：`T1_GAP = mean( r_on,t | r_id,t-1 大跌, N=20 )`（大跌日后的平均隔夜跳空），度量"T+1 无法当日止损"导致的隔夜风险补偿。
- 数据：日线开/收盘。T 收盘可算：是。
- 来源：T+1 制度与交易行为研究（[中国A股涨跌停交易制度与投资者处置效应（金融研究）](http://www.jryj.org.cn/CN/abstract/abstract1373.shtml)）；周内效应中 T+1 机制解释（周四抛压）。
- 证据：T+1 使日内亏损无法止损，恐慌延后至次日开盘释放，形成可预测的隔夜-日内收益结构。
- 失效风险：属间接机制变量，单独 IC 弱，建议作交互项。

### 12.5 DISPOSAL —— 处置效应代理（浮亏持有压力）
- 公式：`DISPOSAL = C_T / VWAP_{60} - 1`（现价相对 60 日持仓成本均价的偏离，VWAP 用成交额/成交量近似）。
- 数据：日线成交额 + 成交量。T 收盘可算：是。
- 来源：处置效应文献（Odean 1998）；A 股证据见 [中国A股涨跌停交易制度与投资者处置效应（金融研究）](http://www.jryj.org.cn/CN/abstract/abstract1373.shtml)：散户在浮亏时惜售、涨跌停制度强化处置效应。
- 证据：现价深低于持仓成本 → 抛压衰竭（惜售）但同时反弹至成本区附近遇解套抛压；该变量直接给出反弹的阻力位结构。
- 失效风险：VWAP 近似持仓成本忽略换手（高换手股票成本锚更接近现价），建议用换手衰减加权 VWAP 改进。

### 12.6 ST_FLAG —— ST/退市风险标记
- 公式：ST/*ST/退市整理期 dummy；上市天数 < 250 的次新股 dummy（两个特征）。
- 数据：证券状态/上市日期。T 收盘可算：是。
- 来源：A 股壳价值与 ST 制度研究（见 [Liu, Stambaugh & Wei 2019 CH-3 构建中对小市值壳价值的处理](http://mp.weixin.qq.com/s?__biz=MzU3MDY2ODU3Mg==&mid=2247500582)）。
- 证据：ST 股波动结构、涨跌停幅度（±5%）、流动性均与正常股不同，必须单独建模或剔除；次新股（开板后）有独特的反转/弹性结构。
- 失效风险：状态变更需用 T 时点历史状态，避免前视。

---

## 家族 13：技术指标的学术化特征（4 个）

> 学术证据：[技术指标在中国市场的预测准确性一致高于美国（8/8 指标，p≪0.05）](https://escholarship.org/content/qt1p90t92h/qt1p90t92h.pdf)，机制为散户技术交易行为的自我强化；[MACD 信号对 19% 的中国个股始终预测正确](https://escholarship.org/content/qt1p90t92h/qt1p90t92h.pdf)。
> 定位：不是方向信号，而是把事件形态参数化，供排序模型学习"什么样的底背离更容易成功"。

### 13.1 MACD 状态参数
- 公式：`DIF = EMA(C,12) - EMA(C,26)`；`DEA = EMA(DIF,9)`；`MACD_HIST = 2×(DIF-DEA)`；输出 `DIF/C`（归一化）、`DIF 斜率`、背离跨度（前低与现低的距离天数）、底背离级别（DIF 抬升幅度 / 价格下探幅度）。
- 数据：日线收盘价。T 收盘可算：是。
- 来源：Appel MACD 原始定义；中国有效性证据上引。
- 证据：MACD 在多空对比研究中准确率高于 RSI（约 80% vs 56% 的引证，见 [MACD 效率检验综述](https://www.impactjournals.us/download/archives/2-78-1495887472-11.Man-Testing%20the%20efficiency%20of%20oscillators%20in%20Indian%20stock%20market%20_2_.pdf)）。
- 失效风险：参数（12,26,9）固化；多次背离（二重/三重底）与单次背离的成功率不同，需显式编码背离次数。

### 13.2 RSI 状态参数
- 公式：`RSI(14) = 100 - 100/(1 + 平均上涨/平均下跌)`；输出 RSI 水平、RSI 与价格背离度、超卖持续天数（RSI<30 的连续天数）。
- 数据：日线收盘价。T 收盘可算：是。
- 来源：Wilder (1978)；中国有效性上引。
- 证据：超卖持续天数比 RSI 水平本身信息量大（钝化现象）；底背离中 RSI 二次探底抬高是标准确认条件。
- 失效风险：强趋势中 RSI 长期钝化失效。

### 13.3 BOLL 位置
- 公式：`PCTB = (C_T - MA20) / (2×std(C,20))`（%B 指标），及带宽 `BW = 4×std(C,20)/MA20`。
- 数据：日线收盘价。T 收盘可算：是。
- 来源：Bollinger Bands；[Bayesian Predictive Score Analysis of RSI, MACD, and Bollinger Bands (2025)](https://www.harbinengineeringjournal.com/index.php/journal/article/view/4798)。
- 证据：%B 是标准化的超跌深度（与 VOL 交互），带宽收缩后突破是高胜率形态。
- 失效风险：与 RET20/VOL20 信息重叠，建议只保留 %B。

### 13.4 KDJ / 均线乖离
- 公式：`BIAS20 = C_T / MA(C,20) - 1`（乖离率）；`KDJ_K, KDJ_D` 按 RSV(9) 递推。
- 数据：日线 OHLC。T 收盘可算：是。
- 来源：中国有效性上引（KDJ 为 8 个被测指标之一）；乖离率即标准化反转因子。
- 证据：BIAS 与 RET20 高度同源（差异仅在均值基准），深乖离是 A 股短线反弹的标准触发条件。
- 失效风险：与 1.1/1.2 冗余，建模时保留其一。

---

## 附：汇总表

| 家族 | 数量 | 特征编号 |
|---|---|---|
| 1 动量/反转 | 8 | 1.1-1.8 |
| 2 波动率 | 7 | 2.1-2.7 |
| 3 流动性/换手 | 8 | 3.1-3.8 |
| 4 量价相关 | 4 | 4.1-4.4 |
| 5 量价背离 | 3 | 5.1-5.3 |
| 6 波动率结构 | 4 | 6.1-6.4 |
| 7 偏度/高阶矩 | 6 | 7.1-7.6 |
| 8 日历效应 | 5 | 8.1-8.5 |
| 9 市场状态 | 6 | 9.1-9.6 |
| 10 行业相对强度 | 4 | 10.1-10.4 |
| 11 资金流代理 | 5 | 11.1-11.5 |
| 12 A股特有制度 | 6 | 12.1-12.6 |
| 13 技术指标参数化 | 4 | 13.1-13.4 |
| **合计** | **70** | |

全部 70 个特征均满足"信号日 T 收盘后可算"的硬约束；其中 8 个需分钟级数据（4.3、6.4、11.1、11.3、11.4 及 4.3 的三维拆分）、1 个需分单数据（11.2）、1 个需关联关系数据（10.4），其余 60 个仅用日线 OHLCV/成交额/股本/日历即可计算。

## 实施优先级备注

1. 第一批（日线可算、A 股证据最强）：1.1、1.2、1.5、3.1、3.3、2.4、2.1、7.1、7.3、12.1、12.3、12.5、9.1、9.4、13.1。
2. 注意机械相关：标签含 ATR，2.1/2.7/2.4 与标签可达性机械相关，必须入模但解读时区分"alpha"与"标签几何"。
3. 冗余对（建模时二选一或正交化）：1.1↔13.4、2.4↔2.1、3.1↔3.4↔3.7、7.1↔12.1、1.6↔10.3、8.1↔8.5。
4. 前视偏差高危点：行业/概念成分（10.x）用 T 时点历史成分；ST 状态（12.6）用历史状态；涨跌停规则按历史时期分段（12.x）。
