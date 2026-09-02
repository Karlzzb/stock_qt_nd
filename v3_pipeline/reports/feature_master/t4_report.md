# T4 新特征实现入池报告（issue #24）

日期：2026-09-02
产出：v3_pipeline/reports/feature_master/cache/t3_snapshot_{main,backup}.parquet（事件日快照）+ t3_results.json（覆盖与抽检台账）+ 更新后的 master_{main,backup}.parquet / master_dictionary.csv / master_results.json
构建：v3_pipeline/scripts/build_t3_features.py（库 v3_pipeline/src/t3_features.py）；主表并入走 build_feature_master.py（库 feature_master.py 扩展来源 4）
单元测试：v3_pipeline/tests/test_t3_features.py（24 例，合成数据已知值断言，无磁盘依赖）；全套件（v3_pipeline/tests + 根 tests/）325 例全绿。

## 结论

**T3 预登记清单 30 条新特征全部实现（落成 41 个数值列），并入主表后两池各 2071 列（11 元数据 + 2060 特征）；泄漏排除与快照一致性两项验收断言在合并后主表上全部通过。**

## 实现范围与列构成

30 条清单特征落为 41 列：30 条主列 + 5 条清单内伴随列（VOLUME_RATIO_MA5、MKT_LIMITUP_PREM_MA5、MKT_NH_NL_RATIO、DAYS_BELOW_PAR、TOUCH_FAIL_DEPTH）+ 6 条口径/标记列（LIMITUP_SEALED_SRC、CONSEC_SUSP_GAP、RESID_MOM60_MKSRC、RESID_MOM60_INDNA 等，同为 T 时点可得的因果量，供模型显式区分口径段）。
中文全名词典 T3_CN 覆盖全部 41 列（命名用全称纪律），master_dictionary.csv 中 s4 来源列中文名覆盖率 100%。

分族：换手/流动性 6 列、规模/估值 4 列、涨跌停个股级 9 列、市场情绪/宽度 9 列、行业/风格/制度 13 列。

## 关键口径与数据事实（实现期实锤，已登记于库模块文档）

- 涨跌停判定双口径：2007-01-04 起用 stk_limit 精确价（容差 0.005 元），此前用 pct_chg 阈值近似（主板 10%、ST/*ST 5%，容差 limit_pct−0.5 个百分点）；1996-12-16 前一律 NaN。
- 上市未满 5 个交易日的个股：涨跌停标记 NaN，且不进入任何横截面分母。
- ST_STATUS 由 namechange 名称区间重建（3=退、2=*ST、1=ST、0=正常），行业由 sw_index_member（申万一级、SW2021）in/out 区间重建——T 时点区间查找，无快照回填。
- 市场日收益拼接：2002-01-04 起 000300.pct_chg，此前 daily/000001.SH（旧 schema 升序重排后 close 自差分），RESID_MOM60_MKSRC 逐行标记口径段（2=纯 300、1=混合、0=纯上证）。
- 复权口径：跨日价格比较一律 pct_chg 链 CF = cumprod(1+pct_chg/100)，只在同股同窗内比较。
- pb/pe_ttm 从不存负值（亏损股 NaN，约占截面 24%）；dv_ttm 无分红股为 NaN 而非 0，显式 NaN→0 映射（"无分红/缺失"不可分，已登记）。
- min_periods 口径（清单未写死者）：滚动 20/21 窗 ≥15、252 窗 ≥126、500 窗 ≥250、750 窗 ≥250、250 滚动矩 ≥126、60 日残差窗 ≥45 有效日、5 日平滑 ≥3、20 日动量窗 NaN 天数 >5 记 NaN。

## 覆盖率与分布体检（主池 8154 事件日快照）

| 列/族 | NaN 率（主/备） | 解读 |
|---|---|---|
| 换手/规模族（TURN_F20 等） | 0.5–0.6% / 0.4–0.5% | 仅极早期数据缺口 |
| ABN_TURN_21_252 | 1.6% / 3.4% | 252 窗 min_periods 口径 |
| BP_IND_Z | 11.3% / 16.8% | 行业未归属 + pb NaN + <10 成分行业门槛 |
| EP_TSRANK_500 | 29.5% / 30.7% | ≈亏损股 24%（pe_ttm 不存负值）+ 次新，与评审预期一致 |
| DV_TTM | 0% / 0% | NaN→0 显式映射后全覆盖 |
| 涨跌停族 | 5.9% / 5.7% | 集中为 1996-12-16 前早期事件（制度外记 NaN） |
| TOUCH_FAIL_DEPTH / DOWNTOUCH_RECOVER | 97.4% / 84.6%（备 99.7%） | 结构性缺失：仅在触发日有定义 |
| 市场宽度族（MKT_*） | 0–2.2% / 0–3.3% | 早期截面过薄段 |
| 行业族（IND_*） | 10.4% / 15.6% | 申万一级未归属事件 |
| RESID_MOM60 | 0.02% / 0.04% | 残差窗有效日门槛 |
| STYLE_GV_RS60 | 4.2% / 21.9% | 399006 序列起点前记 NaN（备池早期事件多） |
| ST_STATUS / PAR_VALUE_GAP / DAYS_BELOW_PAR / LIST_AGE | 0% / 0% | 区间/日历重建全覆盖 |

触发率与口径分布（主池）：
收盘封涨停判定 91.9% 走精确价、2.2% 阈值近似、5.9% 制度外缺失。
一字涨停触发 0.25%、炸板 0.85%、跌停撬开 1.34%（备池 0.26%，与两池事件结构差异方向一致）。
ST 分布 95.97% 正常 / 1.52% ST / 2.51% *ST。
残差口径 98.9% 纯沪深300 / 1.0% 纯上证。
连板含停牌跳空仅 0.05%。

## 验收断言（issue #24 AC）

1. **评审通过的新特征全部实现并入主表**：PASS。
   30/30 条实现；s4 快照（8154×43 / 36986×43，含键；两池事件键缺失均为 0）经 build_feature_master.py 并入，AC1 硬断言四来源全部在场。
   泄漏排除扫描对 41 个 T3 列零命中（命中仅既有 4 个 rank_* 衍生列）；同式去重 2443→2060 剔 383 列，其中 T3 仅 3 列：PAR_VALUE_GAP（ρ=1.0 vs s1 LOG_PRICE，同构不同参数化）、TOUCH_FAIL_DEPTH（ρ=0.99919 vs s1 SNAP_H）、DOWNTOUCH_RECOVER（ρ=−0.99918 vs s1 SNAP_L）——后两者与 s1 快照高低点列是同一日内价格结构的两种参数化，按保留优先级 s1>s4 剔除，口径信息无损失。
   最终主表：主池 8154×2071、备池 36986×2071；T3 净入池 38 列。
2. **泄漏排除与快照一致性断言在合并后主表上通过**：PASS。
   泄漏：assert_no_leakage 在合并后主表列集上通过（构建脚本硬断言）。
   快照一致性两层证据：
   (a) 构建侧截断前缀重算抽检 8 单元格（确定性种子 20260902，跨 2008–2026 与主板/科创/创业/深沪），全历史值与仅用 ≤T 数据前缀重算值逐位一致（rtol 1e-9），0 不一致。
   (b) 主表构建末端对两池各 4 个新鲜事件单元格做主表 T3 列 vs 当场截断重算对拍，0 不一致。
   时点一致性的构造性保证：全部算子为 rolling/shift/cumprod 回看或 T 时点横截面聚合；交易日历、ST/行业区间、指数序列经截断不变量 ctx 传入，与面板截断解耦（库模块文档 §纪律）。

## 过程中发现的历史问题（根 tests/ 遗留故障，本次一并修复）

- tests/fixtures/daily/ 三个 fixture parquet 在仓库瘦身重写历史后从未入库，导致根 tests/ 41 个收集/固件错误 + 2 个失败；已用 scripts/generate_test_fixtures.py 确定性重生成（000001.SZ 2021 切片、000004.SZ 退市切片、div_trigger 合成序列），回归基线逐位匹配，41+2 全转绿；fixture 已按生成脚本约定随本次提交入库（56K，可离线复现）。
- src/tinyshare_auth.py 存在硬编码默认 token 兜底，使"token 缺失抛 EnvironmentError"契约不可达（test_env_layer 2 例失败），且属凭据入源码的安全隐患；已删除兜底，契约恢复。
- tests/test_env_layer.py 空 token 用例未覆盖 TTSHARE_TOKEN 别名（ttshare 迁移期引入，真实 .env 含该别名）导致误失败；用例已改为两个别名同时置空，语义（空环境变量优先于 .env）不变。
- 修复后全套件 325 例全绿（v3_pipeline/tests 34 + 根 tests/ 291）。

## 对后续票据的约束

- ⑤ 模型搭建与 ⑥ 标签赛：特征列 = 主表 2060 列（含 T3 净入池 38 列）；标签独立计算，禁入主表。
- T3 口径/标记列（*_SRC、*_MKSRC、*_INDNA、CONSEC_SUSP_GAP）为因果可得的元信息列，模型可用可弃，但不得当作标签代理误读。
- EP_TSRANK_500 的 NaN 含"亏损股"语义（pe_ttm 不存负值），下游如做缺失指示变量需知悉该口径。

## 独立复核记录

证伪式独立复核（全部自算、不信任构建脚本自报）七项全部 PASS：
A 行与键（主池 8158−4 指数伪股=8154、备池 37012−26=36986，键集合双向零差集）；
B 手工重算（自选 6 事件跨近似段/精确段/科创年代，TURN_F20/LN_FREE_MV/LIMITUP_SEALED_EXACT/ST_STATUS 共 24 值语义一致，含 *ST 近似段 5% 阈值阳性验证）；
C 泄漏扫描（独立转写 17 条排除模式扫 2071 列名，0 命中）；
D 去重（3 个 T3 剔除对 ρ 逐位吻合且 ≥0.999；保留列 300 随机对最大 |ρ|=0.9558）；
E 时点一致性（独立实现 ≤T 前缀重算 LIST_AGE/ABN_TURN_21_252/CONSEC_LIMITUP 三特征，与主表逐位相等）；
F 词典完整性（s4 行 41 条中文名无空，kept 38 列与 T3_COLUMNS 减 3 剔除精确匹配）；
G 报告交叉核对（6 列 NaN 率重算与报告数字全吻合）。
复核代码留存于 /tmp/t4_verify/（一次性对抗脚本，不入库）。

复核口径发现一项：主表特征值为 float32 中间精度（build_panel 文档化降精度），逐位复现须按 float32 约定重算，纯 float64 复算偏差 ≤9.4e-8，登记不改。

两轴代码评审（标准轴 + 规格轴）发现与处置：
1. 规格轴：RESID_MOM60_INDNA 原按 T 日行业归属判定，未忠实于预登记 #25"行业 NaN 退化为仅减市场收益并置缺失标记"的窗口语义——已改为窗口 [T−64, T−5] 内任一行业缺失日即置 1（退化日指示窗口 max、同 shift、同 min_periods），补窗口级单元测试 test_resid_mom_indna_window_level，快照与主表已全部重建重断言（主池退化标记率 10.58%）。
2. 规格轴：驱动文档承诺"快照键覆盖缺失计数落盘"未实现——已实现并记入 t3_results.json（两池缺失均 0）。
3. 标准轴：T3/T4 术语跨文件混用——库模块文档已加命名口径段（T3=候选清单所属步骤 #23，T4=本实现票据 #24）。
4. 标准轴：_assign_interval_num/_assign_interval_str 25 行重复——合并为单一 _assign_interval(default, dtype)。
5. 标准轴：_add_limit_features 四段同构"精确/近似"双分支填充——提取 _exact_approx_flag 助手收敛。
6. 标准轴：coverage_report 死参数 feat_full——已删除。
7. 标准轴：报告多处单物理行多句——已按每句一行拆排。
8. 登记不改：两驱动脚本的抽检比较核（prefix_spot_check vs spot_check_s4）同构，但采样与报告逻辑各异、共享核心仅 5 行逐位比较，提取共享助手的收益不抵跨脚本耦合；ctx 六键裸 dict 与仓库既有风格一致。

评审修复后快照、主表、词典全部重建并重过断言（泄漏排除 T3 零命中；前缀抽检 8 单元 + s4 末端抽检 4+4 单元零不一致），全套件 325 例全绿。
