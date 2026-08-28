for year in {2010..2024}; do
 for month in {1..12}; do
    uv run python scripts/run_feature_pipeline_incremental.py --year $year --month $month --workers 24 --batch-size 20
 done
done
