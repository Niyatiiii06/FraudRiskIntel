# FraudRiskIntel — Fraud Detection & Risk Intelligence

**The product feature:** payment risk tools such as Stripe Radar score each payment, show the
main risk factors, and let merchants choose a preset risk appetite (for example "maximize
protection" vs "maximize revenue"). The weakness for a learner or a smaller team: the model,
its features and its trade-off are closed, so you can't inspect or reproduce how a score or
a block decision was made.

**This project:** an open, inspectable version of that workflow for mobile-money transfers.
Every transaction gets a fraud probability, a block/allow decision set from explicit business
costs you choose (cost of a missed fraud vs a false alarm), and quantified per-feature reasons
(SHAP). It is measured against a tuned rules baseline, the standard starting approach, on a
fixed later time period. It makes no claim about any real product's accuracy.

Built on the PaySim mobile-money dataset (6.3M synthetic transactions): SQL feature engineering
in DuckDB, time-safe graph features, LightGBM with a time-based split, and SHAP explanations.

## Results (test set: last ~15% of the month, 408,077 transactions, used once)

| Approach | Frauds caught | Precision | Fraud money caught | Money lost* |
|---|---|---|---|---|
| Flag nothing | 0 / 4,010 | – | 0% | 6.32B |
| Rules baseline (tuned on validation) | 1,053 (26%) | 0.26 | 60% | 2.53B |
| **LightGBM model** | **2,700 (67%)** | **0.30** | **92%** | **0.52B** |

\*Missed fraud amount + a review fee of 100 per alarm (PaySim currency; the fee is an assumption).

- The model loses **79.6% less money** than the tuned rules baseline on this test set.
- PR-AUC 0.35 vs 0.0098 for random. ROC-AUC 0.94.
- Ablation (test PR-AUC): base features 0.350, + graph features 0.352, + balance features 0.9999.
  The ~1.0 scores often reported on PaySim come from the simulator's balance columns,
  so the final model leaves them out.

## Limitations
- PaySim is synthetic; results show the method, not real-world fraud rates.
- Graph features added almost nothing here, because PaySim never reuses fraud accounts.
- SHAP shows the model leans on hour of day (PaySim's night hours are mostly fraud), which
  causes confident false alarms at night. Not fixed, to keep the test set untouched.
- Senders appear only once, so frauds that look normal from the receiver's side are missed.

> **Status:** steps 1–7 done; API, CI/Docker and demo still to come.

## Roadmap
- [x] 1. Load PaySim into DuckDB
- [x] 2. SQL feature engineering (receiver velocity, account history, balance mismatches)
- [x] 3. Graph features (time-safe connected components, links to known fraud)
- [x] 4. Models with time-based train/val/test split, plus a feature-group ablation (validation only)
- [x] 5. Class weights, cost-based threshold, single test-set evaluation, comparison with a tuned rules baseline
- [x] 6. SHAP explanations (global + per transaction)
- [x] 7. MLflow experiment tracking (local)
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
