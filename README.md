# FraudRiskIntel — Fraud Detection & Risk Intelligence

End-to-end fraud detection on the PaySim mobile-money dataset (6.3M synthetic transactions):
SQL feature engineering in DuckDB, transaction-graph features, gradient-boosted models with a
time-based split, cost-based decision thresholds, SHAP explanations, MLflow tracking, and a
FastAPI scoring service with drift monitoring.

> **Status:** in progress — steps 1–2 (data loading, SQL features) done. Results will be added only after evaluation is run.

## Roadmap
- [x] 1. Load PaySim into DuckDB
- [x] 2. SQL feature engineering (receiver velocity, account history, balance mismatches)
- [ ] 3. Graph features (NetworkX)
- [ ] 4. Models with time-based train/val/test split, plus a with/without balance-features ablation
- [ ] 5. Imbalance handling, PR-AUC, cost-based threshold
- [ ] 6. SHAP explanations (global + per transaction)
- [ ] 7. MLflow experiment tracking
- [ ] 8. FastAPI `/score` service + PSI drift check
- [ ] 9. Tests, CI, Docker, Streamlit demo
- [ ] 10. Final results

## Key design choices
- **Scored population:** only TRANSFER and CASH_OUT (the only types with fraud); history features still use all types.
- **Receiver-side history:** 99.85% of senders appear only once, so velocity features are built on the receiving account.
- **No leakage in features:** history windows only see strictly earlier hours.

## Data
PaySim synthetic dataset by E. A. Lopez-Rojas et al., from Kaggle (CC BY-SA 4.0). Not stored in this repo.

Setup and commands: see [SETUP.md](SETUP.md).
