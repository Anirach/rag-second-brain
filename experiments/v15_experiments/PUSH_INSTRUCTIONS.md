# GitHub Push Instructions

The code has been committed locally but needs credentials to push to GitHub.

## Files Created

All files are in `/tmp/rag-second-brain/experiments/`:

1. `learned_gating.py` - Sigmoid/Softmax gating implementations
2. `statistics.py` - Bootstrap CI, paired t-tests
3. `evaluate.py` - EM/F1 metrics, query classification
4. `run_v15_experiments.py` - Full runner (requires numpy/scipy)
5. `run_v15_experiments_pure.py` - Pure Python version
6. `V15_EXPERIMENT_SUMMARY.md` - Results summary
7. `results/v15_results.json` - JSON results
8. Updated `README.md` and `results/results.json`

## To Push Manually

```bash
cd /tmp/rag-second-brain

# If using HTTPS with token:
git remote set-url origin https://<TOKEN>@github.com/Anirach/rag-second-brain.git
git push origin main

# If using SSH:
git remote set-url origin git@github.com:Anirach/rag-second-brain.git
git push origin main
```

## Or Apply Patch to Fresh Clone

```bash
git clone https://github.com/Anirach/rag-second-brain.git
cd rag-second-brain
git apply /tmp/0001-V15-experiments-Learned-gating-statistics-EM-F1-metr.patch
git push origin main
```

## Commit Message

```
V15 experiments: Learned gating, statistics, EM/F1 metrics

- Added learned_gating.py: Sigmoid and Softmax gating modules
- Added statistics.py: Bootstrap CI (1000 resamples), paired t-tests
- Added evaluate.py: EM/F1 metrics, query type classification
- Added run_v15_experiments.py and pure Python version
- Fixed 8.5% → 8.4% inconsistency in results
- Created V15_EXPERIMENT_SUMMARY.md with all results

Key findings:
- Sigmoid gating: R@10 = 0.923 (+18.4% vs RRF, p < 0.001)
- Softmax gating: R@10 = 0.911 (+16.9% vs RRF)
- Cohen's d = 0.475 (medium effect size)

All experiments use fixed random seed (42) for reproducibility.
```
