# 市场特征恢复后的性能对比

生成时间: 2026-08-28 12:36:38.062354

## 问题背景

之前的训练缺少 13 个市场特征（sh_/sz_ 前缀）和 1 个背离特征（divergence_magnitude），
原因是数据加载 bug 导致指数数据未正确合并。

## 修复内容

1. run_feature_pipeline.py: 指数文件列名 volume 不是 vol
2. feature_pipeline_v2.py load_price_data: 自动加载指数数据
3. _calculate_single_index_features: 时间戳类型转换
4. calculate_macd_percentile_vectorized: 窗口不足返回 NaN

## 恢复后的 Rank Metrics

验证集: 2022-01-01 至 2025-07-31

```
            label        IC      ICIR  top5_excess tradable
future_return_10d  0.034249  0.162305     0.000073        ❌
future_return_30d  0.028178  0.139098    -0.005780        ❌
 future_return_5d  0.016131  0.083138     0.001905        ❌
future_return_25d  0.005780  0.027132    -0.009716        ❌
future_return_20d -0.014446 -0.068449    -0.012766        ❌
future_return_15d -0.019681 -0.096174    -0.013197        ❌
 future_return_3d -0.023739 -0.115047    -0.002327        ❌
```

可交易标准: ICIR > 1.5 AND top5_excess > 2%

**结果: ⚠️ 暂无 label 达到可交易阈值**
