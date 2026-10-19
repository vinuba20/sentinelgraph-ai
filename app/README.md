# SentinelGraph dashboard

After a training run, start the dashboard from the repository root:

```bash
streamlit run app/dashboard.py
```

Enter the experiment directory (default: `artifacts/latest`). The dashboard shows the same-test-set comparison, test-flow prediction summaries, global feature importance, and a local approximation of why an uploaded flow was flagged.

Predictions are experimental and must not be used as a live security guarantee.
