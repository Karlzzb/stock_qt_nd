#!/usr/bin/env python3
"""
对比修复前后的特征集变化
"""
import json
from pathlib import Path

# 用户提供的158个重要特征
USER_FEATURES = [
    'close', 'volume', 'turn', 'amount', 'high', 'low', 'open',
    'close_d0.4_0', 'close_d0.4_1', 'close_d0.4_2', 'close_d0.4_3', 'close_d0.4_4', 'close_d0.4_5',
    'close_wavelet_a3', 'close_wavelet_d1', 'close_wavelet_d2', 'close_wavelet_d3',
    'atr_14', 'atr_21',
    'boll_down', 'boll_mid', 'boll_pct', 'boll_pct_rankpct', 'boll_up', 'boll_width', 'boll_width_rankpct',
    'cci_10', 'cci_20',
    'dma_diff', 'dma_mean',
    'kama_10',
    'kdjk', 'kdjd', 'kdjj', 'kdjk_rankpct',
    'macd', 'macd_signal', 'macd_diff',
    'macd_percentile', 'macd_percentile_rankpct',
    'psy_12',
    'roc_5',
    'rsi_6', 'rsi_12',
    'trix', 'trix_signal',
    'vr_26',
    'willr_14',
    'adx_14',
    'aroon_up', 'aroon_down',
    'dm_p', 'dm_n',
    'pdi', 'mdi',
    'volume_delta_5', 'volume_delta_10', 'volume_delta_20',
    'volume_delta_pct_20',
    'volume_ratio',
    'volume_ma_5', 'volume_ma_10', 'volume_ma_20',
    'obv', 'obv_ema_20',
    'pvt',
    'vwap',
    'high_low_spread',
    'close_open_ratio',
    'volatility_5', 'volatility_10', 'volatility_20',
    'volatility_20_rankpct',
    'avg_price',
    'price_efficiency_10', 'price_efficiency_20',
    'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20',
    'momentum_20_rankpct',
    'return_skew_20',
    'return_autocorr_5',
    'sma_5', 'sma_10', 'sma_20', 'sma_60',
    'ema_5', 'ema_10', 'ema_20', 'ema_60',
    'ema_10_20_cross',
    'price_to_sma20', 'price_to_sma60',
    'sma5_sma20_ratio',
    'williams_r_14',
    'intraday_volatility',
    'intraday_momentum',
    'high_low_range',
    'wick_ratio',
    'body_ratio',
    'volume_price_corr_10', 'volume_price_corr_20',
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d',
    'ret_vol_5d', 'ret_vol_10d', 'ret_vol_20d',
    'turnover_volatility_10', 'turnover_volatility_20',
    'ret_skew_20',
    'rank_return_5', 'rank_return_10', 'rank_return_20',
    'rank_volume_5', 'rank_volume_10', 'rank_volume_20',
    'amihud_illiq_20',
    'rs_strength_120',
    'max_draw_down_20', 'max_draw_down_60',
    'up_down_vol_ratio_20',
    'hurst_20',
    'divergence_magnitude', 'divergence_strength',
    'kmeans2_label', 'kmeans3_label', 'kmeans4_label', 'kmeans5_label',
    'sh_price_change', 'sh_amplitude', 'sh_volume_ratio',
    'sh_sma_5', 'sh_sma_10', 'sh_sma_20', 'sh_sma_60',
    'sh_ema_5', 'sh_ema_10', 'sh_ema_20', 'sh_ema_60',
    'sh_rsi_6', 'sh_rsi_12',
    'sh_macd', 'sh_macd_signal', 'sh_macd_diff',
    'sh_boll_pct', 'sh_boll_width',
    'sh_atr_14',
    'sh_ret_1d', 'sh_ret_5d', 'sh_ret_10d', 'sh_ret_20d',
    'sh_volatility_20',
    'sh_momentum_10', 'sh_momentum_20',
    'sz_price_change', 'sz_amplitude', 'sz_volume_ratio',
    'sz_sma_5', 'sz_sma_10', 'sz_sma_20', 'sz_sma_60',
    'sz_ema_5', 'sz_ema_10', 'sz_ema_20',
    'sz_rsi_6', 'sz_rsi_12',
    'sz_macd', 'sz_macd_signal', 'sz_macd_diff',
    'sz_boll_pct', 'sz_boll_width',
    'sz_atr_14',
    'sz_ret_1d', 'sz_ret_5d', 'sz_ret_10d', 'sz_ret_20d',
]

def main():
    # 加载当前的特征集（修复前的552特征）
    old_features_path = Path('models/single_2021-12-31_2022-01-01_2025-07-31_poolfull/lgbm_features.json')

    if not old_features_path.exists():
        print(f"❌ 未找到旧特征集: {old_features_path}")
        return

    with open(old_features_path) as f:
        old_features = json.load(f)

    print("="*70)
    print("特征集对比分析")
    print("="*70)

    # 转为集合
    user_set = set(USER_FEATURES)
    old_set = set(old_features)

    # 分析用户特征的状态
    present = user_set & old_set
    missing = user_set - old_set

    print(f"\n用户提供的重要特征: {len(user_set)} 个")
    print(f"修复前特征池: {len(old_set)} 个")
    print(f"用户特征中存在的: {len(present)} 个 ({len(present)/len(user_set)*100:.1f}%)")
    print(f"用户特征中缺失的: {len(missing)} 个 ({len(missing)/len(user_set)*100:.1f}%)")

    # 分类缺失特征
    market_missing = [f for f in missing if f.startswith(('sh_', 'sz_'))]
    divergence_missing = [f for f in missing if 'divergence' in f]
    wavelet_missing = [f for f in missing if 'wavelet' in f]
    d04_missing = [f for f in missing if 'close_d0.4' in f]
    other_missing = [f for f in missing if f not in market_missing + divergence_missing + wavelet_missing + d04_missing]

    print("\n" + "="*70)
    print("缺失特征分类")
    print("="*70)

    if market_missing:
        print(f"\n市场特征 (sh_/sz_): {len(market_missing)} 个")
        for f in sorted(market_missing):
            print(f"  - {f}")

    if divergence_missing:
        print(f"\n背离特征: {len(divergence_missing)} 个")
        for f in divergence_missing:
            print(f"  - {f}")

    if wavelet_missing:
        print(f"\nWavelet 特征: {len(wavelet_missing)} 个")
        for f in wavelet_missing:
            print(f"  - {f}")

    if d04_missing:
        print(f"\nD0.4 特征: {len(d04_missing)} 个")
        for f in d04_missing:
            print(f"  - {f}")

    if other_missing:
        print(f"\n其他缺失: {len(other_missing)} 个")
        for f in sorted(other_missing):
            print(f"  - {f}")

    # 估算修复后的特征池大小
    expected_recovery = len(market_missing) + len(divergence_missing)
    print("\n" + "="*70)
    print("修复预期")
    print("="*70)
    print(f"\n本次修复可恢复: {expected_recovery} 个特征")
    print(f"  - 市场特征: {len(market_missing)} 个")
    print(f"  - 背离特征: {len(divergence_missing)} 个")
    print(f"\n修复后预期特征池: {len(old_set) + expected_recovery} 个")
    print(f"用户特征覆盖率: {(len(present) + expected_recovery) / len(user_set) * 100:.1f}%")

    # 说明无法恢复的特征
    cannot_recover = len(wavelet_missing) + len(d04_missing) + len(other_missing)
    if cannot_recover > 0:
        print(f"\n无法恢复的特征: {cannot_recover} 个")
        print("原因:")
        if wavelet_missing:
            print(f"  - Wavelet 特征 ({len(wavelet_missing)}个): 未在当前特征工程中实现")
        if d04_missing:
            print(f"  - D0.4 特征 ({len(d04_missing)}个): 未在当前特征工程中实现")
        if other_missing:
            print(f"  - 其他 ({len(other_missing)}个): 可能已重命名或删除")

if __name__ == '__main__':
    main()
