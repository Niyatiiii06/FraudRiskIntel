"""Fraud scoring API.

Run:  uvicorn fraud.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import os
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from fraud.config import MODEL_PATH
from fraud.serving import load_bundle, score_transaction

app = FastAPI(title="FraudRiskIntel", version="1.0")


class Transaction(BaseModel):
    step: int = Field(ge=1, description="hour of the simulation (1 step = 1 hour)")
    txn_type: Literal["TRANSFER", "CASH_OUT"]
    amount: float = Field(ge=0)
    # receiver's history before this hour; leave out for a brand-new receiver
    dest_txns_prev_24h: int = Field(0, ge=0)
    dest_amount_prev_24h: float = Field(0, ge=0)
    dest_txns_prev_all: int = Field(0, ge=0)
    dest_avg_amount_prev: float | None = None
    dest_hours_since_prev: int | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "step": 400,
                "txn_type": "TRANSFER",
                "amount": 850000,
                "dest_txns_prev_24h": 0,
                "dest_amount_prev_24h": 0,
                "dest_txns_prev_all": 0,
            }
        }
    }


class Reason(BaseModel):
    feature: str
    description: str
    value: str
    impact: float


class ScoreResponse(BaseModel):
    fraud_probability: float
    decision: Literal["review", "allow"]
    threshold: float
    top_reasons: list[Reason]


@lru_cache
def get_bundle():
    return load_bundle(os.environ.get("FRAUD_MODEL_PATH", MODEL_PATH))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction, bundle=Depends(get_bundle)):
    return score_transaction(bundle, txn.model_dump())
