# FraudRiskIntel — Fraud Detection & Risk Intelligence

![CI](https://github.com/Niyatiiii06/FraudRiskIntel/actions/workflows/ci.yml/badge.svg)

**Live demo:** https://fraudriskintel-kakayo83bhmwawv8lfqsty.streamlit.app/

**The product feature:** payment risk tools such as Stripe Radar score each payment, show the
main risk factors, and let merchants choose a preset risk appetite (for example "maximize
protection" vs "maximize revenue"). The weakness for a learner or a smaller team: the model,
its features and its trade-off are closed, so you can't inspect or reproduce how a score or
a block decision was made.

**This project:** an open, inspectable version of that workflow for mobile-money transfers.
Every transaction gets a fraud probability, a review/allow decision set from explicit business
costs (a missed fraud vs a false alarm), and the reasons behind each flag (SHAP). It is measured
against a tuned rules baseline on a later time period. It makes no claim about any real
product's accuracy.

## Results (test set: last ~15% of the month, 408,077 transactions, used once)

| Approach | Frauds caught | Precision | Fraud money caught | Money lost* |
|---|---|---|---|---|
| Flag nothing | 0 / 4,010 | – | 0% | 6.32B |
| Rules baseline (tuned on validation) | 1,053 (26%) | 0.26 | 60% | 2.53B |
| **LightGBM model** | **2,700 (67%)** | **0.30** | **92%** | **0.52B** |

\*Missed fraud amount + a review fee of 100 per alarm (PaySim currency; the fee is an assumption).

- **79.6% less money lost** than the tuned rules baseline on this test set.
- PR-AUC 0.35 (random: 0.0098), ROC-AUC 0.94.
- With PaySim's balance columns the same model reaches PR-AUC 0.9999. Those columns are a
  simulator artifact, so the final model leaves them out.

## How it works
1. **Clues per payment (SQL in DuckDB):** amount, transaction type, hour, and the receiving
   account's history before that hour (is it new, how much does it usually receive).
2. **Learn from the past:** LightGBM trained on the first ~70% of the month.
3. **Test on the future:** chosen on the next ~15%, scored once on the last ~15%. No random split.
4. **Decide by cost:** flag for review if the fraud probability is at least 2.1%, the cut-off
   that minimises cost when a missed fraud costs 50x a false alarm.
5. **Explain:** SHAP gives the top reasons behind every score.
6. **Serve and monitor:** FastAPI `/score` endpoint, PSI drift check, MLflow run tracking,
   Docker image, CI, and a Streamlit demo.

## What the analysis revealed
- **Dataset limits:** PaySim is synthetic and senders appear only once, so the model can only use
  the receiver's history, and graph features added nothing (fraud accounts are never reused).
- **Model behaviour:** SHAP shows heavy reliance on hour of day (PaySim's night traffic is mostly
  fraud): the same transfer scores 1% at 4 pm and 69% at 6 am. It was not changed after seeing test
  results.
- **Drift:** PSI flags a big shift in the 24-hour velocity features because PaySim's volume drops
  over 80% late in the month. One of them is in the model's top 5, so in a real system this would
  trigger retraining on recent data.

## Data
PaySim synthetic dataset by E. A. Lopez-Rojas et al., from Kaggle (CC BY-SA 4.0). Not stored in this repo.

Setup, commands and design details: see [SETUP.md](SETUP.md).
