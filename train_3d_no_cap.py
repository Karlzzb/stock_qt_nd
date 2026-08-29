#!/usr/bin/env python3
"""训练3d horizon模型（无截断数据）"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent

# Train 3d model
config_path = REPO_ROOT / "v3_pipeline/configs/v3_0_0_baseline.yaml"

# Read config and modify to train only 3d
import yaml
with open(config_path) as f:
    config = yaml.safe_load(f)

# Create temp config for 3d only
config['training']['horizons'] = ['3d']
temp_config = REPO_ROOT / "v3_pipeline/configs/temp_3d_only.yaml"
with open(temp_config, 'w') as f:
    yaml.dump(config, f)

print("Training 3d model with clean data...")
result = subprocess.run([
    sys.executable,
    "v3_pipeline/scripts/train_ranking.py",
    "--cache", "v3_pipeline/feature_cache_v3_no_cap.parquet",
    "--output", "v3_pipeline/models/v3_0_3_no_cap",
    "--config", str(temp_config)
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)

print("\n✓ 3d model trained")
