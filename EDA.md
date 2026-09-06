# EDA on Dataset
[Dataset EDA report](https://mehta-a.github.io/dummy-submission/)

### commands to run the individual model evaluation (now in bench.py)
```
cp runs/v2/snapshot-eNNN.pt checkpoint/model.pt
git add checkpoint/model.pt && git commit -m "VAE at NNN epochs" && git push
cd ../amp-grader-demo && rm -rf submission
uv run python scripts/grade_submission.py all ../dummy-submission/ \
    --skip-train --scorer-profile grader --report-out reports/mine_v2.json

uv run python scripts/aggregate_scores.py --reports-dir reports --out reports/leaderboard.json
```

