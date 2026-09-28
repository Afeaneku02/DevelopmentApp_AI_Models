"""Separate, durable usage ledger. Contains no prompts or user-model records.

Reservations are deliberately never refunded, including timeouts/crashes. This
keeps the local experiment ceiling conservative across processes and restarts.
It is an estimated ceiling at versioned prices, not a provider billing limit.
"""
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from src.llm.base import ProviderUnavailable
from src.llm.models import UsageTelemetry

# Standard short-context USD per million tokens; input uses cache-write price
# to conservatively cover both uncached input and cache writes.
PRICES = {"gpt-5.6-luna": (0.25, 1.20), "gpt-5.6-terra": (2.50, 12.0), "gpt-5.6-sol": (5.0, 20.0)}


class BudgetLedger:
    def __init__(self, path: Path):
        self.path = path

    def reserve(self, model: str, input_bound: int, output_bound: int, budget: float, max_calls: int):
        if model not in PRICES:
            raise ProviderUnavailable("model_price_unknown")
        input_rate, output_rate = PRICES[model]
        micros = math.ceil(input_bound * input_rate + output_bound * output_rate)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS mentor_calls (id TEXT PRIMARY KEY, reserved_micros INTEGER NOT NULL, telemetry TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
            db.execute("BEGIN IMMEDIATE")
            used, count = db.execute("SELECT COALESCE(SUM(reserved_micros),0), COUNT(*) FROM mentor_calls").fetchone()
            if count >= max_calls or used + micros > math.floor(budget * 1_000_000):
                raise ProviderUnavailable("budget_exhausted")
            run_id = str(uuid4())
            db.execute("INSERT INTO mentor_calls(id,reserved_micros) VALUES (?,?)", (run_id, micros))
        return run_id, micros / 1_000_000

    def finish(self, run_id: str, telemetry: UsageTelemetry):
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute("UPDATE mentor_calls SET telemetry=? WHERE id=?", (telemetry.model_dump_json(), run_id))

    def summary(self):
        if not self.path.exists():
            return {"calls": 0, "reserved_usd": 0.0, "estimated_usage_usd": 0.0}
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute("SELECT reserved_micros,telemetry FROM mentor_calls").fetchall()
        return {"calls": len(rows), "reserved_usd": sum(r[0] for r in rows)/1_000_000,
                "estimated_usage_usd": sum((json.loads(r[1]).get("estimated_cost_usd") or 0) for r in rows if r[1])}
