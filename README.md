# FraudRiskIntel — Fraud Detection & Risk Intelligence

End-to-end fraud detection on the PaySim mobile-money dataset (6.3M synthetic transactions):
SQL feature engineering in DuckDB, transaction-graph features, gradient-boosted models with a
time-based split, cost-based decision thresholds, SHAP explanations, MLflow tracking, and a
FastAPI scoring service with drift monitoring.

> **Status:** in progress — steps 1–4 done (features, graph features, model comparison on validation). Results will be added only after evaluation is run.

## Roadmap
- [x] 1. Load PaySim into DuckDB
- [x] 2. SQL feature engineering (receiver velocity, account history, balance mismatches)
- [x] 3. Graph features (time-safe connected components, links to known fraud)
- [x] 4. Models with time-based train/val/test split, plus a feature-group ablation (validation only)
- [ ] 5. Imbalance handling, PR-AUC, cost-based threshold
- [ ] 6. SHAP explanations (global + per transaction)
- [ ] 7. MLflow experiment tracking
- [ ] 8. FastAPI `/score` service + PSI drift check
- [ ] 9. Tests, CI, Docker, Streamlit demo
- [ ] 10. Final results

## Key design choices
- **Scored population:** only TRANSFER and CASH_OUT (the only types with fraud); history features still use all types.
- **Receiver-side history:** 99.85% of senders appear only once, so velocity features are built on the receiving account.
- **Locked test set:** models and feature sets are compared on validation only; the test set is used once, at the end.
- **No leakage in features:** history windows only see strictly earlier hours; the transaction graph is grown hour by hour (union-find), and fraud labels count only after a 24-hour confirmation delay.

## Data
PaySim synthetic dataset by E. A. Lopez-Rojas et al., from Kaggle (CC BY-SA 4.0). Not stored in this repo.

Setup and commands: see [SETUP.md](SETUP.md).
