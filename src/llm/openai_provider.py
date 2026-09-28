from __future__ import annotations

import json
import logging
import re
import time

from src.llm.base import ProviderUnavailable
from src.llm.budget import BudgetLedger, PRICES
from src.llm.config import Settings
from src.llm.models import MentorContext, MentorGuidance, ProviderResult, UsageTelemetry
from src.llm.prompts import SYSTEM_PROMPT
from src.llm.validation import validate_guidance


class OpenAIMentorProvider:
    def __init__(self, settings: Settings, *, client=None):
        self.settings = settings
        self._client = client
        self.ledger = BudgetLedger(settings.ledger_path)

    def generate_guidance(self, context: MentorContext) -> ProviderResult:
        cfg = self.settings
        def unavailable(code):
            return ProviderUnavailable(code, UsageTelemetry(provider="openai", model=cfg.model, failure_code=code))
        if not cfg.api_key.get_secret_value():
            raise unavailable("missing_api_key")
        payload = context.model_dump_json()
        schema = MentorGuidance.model_json_schema()
        # UTF-8 bytes conservatively bound text tokens; include instructions,
        # schema and generous protocol overhead, not only the user's payload.
        size = len((payload + SYSTEM_PROMPT + json.dumps(schema)).encode("utf-8"))
        if size > 24000:
            raise unavailable("context_too_large")
        try:
            run_id, reservation = self.ledger.reserve(
                cfg.model, size + 2048, cfg.max_output_tokens, cfg.budget_usd, cfg.max_calls,
            )
        except ProviderUnavailable as exc:
            raise unavailable(exc.code) from None
        except Exception:
            raise unavailable("budget_storage_error") from None

        telemetry = UsageTelemetry(provider="openai", model=cfg.model, operation_id=run_id, reserved_cost_usd=reservation)
        started = time.monotonic()
        error = None
        guidance = None
        try:
            from openai import OpenAI
            # SDK debug logging includes request bodies. Keep it disabled even
            # when OPENAI_LOG=debug is set in the parent shell.
            logging.getLogger("openai").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            # Fixed official endpoint; never honor OPENAI_BASE_URL or route keys
            # to a proxy. No hidden SDK retries: every attempt gets a reservation.
            client = self._client or OpenAI(
                api_key=cfg.api_key.get_secret_value(), base_url="https://api.openai.com/v1",
                timeout=cfg.timeout_seconds, max_retries=0,
            )
            try:
                response = client.responses.create(
                    model=cfg.model, instructions=SYSTEM_PROMPT,
                    input=[{"role": "user", "content": payload}],
                    text={"format": {"type": "json_schema", "name": "mentor_guidance", "strict": True, "schema": schema}},
                    max_output_tokens=cfg.max_output_tokens,
                    reasoning={"effort": "low"}, service_tier="default", store=False,
                )
            finally:
                if self._client is None:
                    client.close()
            request_id = getattr(response, "_request_id", None)
            if not isinstance(request_id, str) or not re.fullmatch(r"req_[A-Za-z0-9_-]{1,100}", request_id):
                request_id = None
            updates = {"request_id": request_id}
            returned_model = getattr(response, "model", None)
            if isinstance(returned_model, str) and re.fullmatch(r"gpt-[a-z0-9.-]{1,80}", returned_model):
                updates["returned_model"] = returned_model
            usage = response.usage
            if usage is not None:
                input_rate, output_rate = PRICES[cfg.model]
                updates.update(
                    input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                    cached_input_tokens=getattr(usage.input_tokens_details, "cached_tokens", None),
                    reasoning_tokens=getattr(usage.output_tokens_details, "reasoning_tokens", None),
                    # Conservative input accounting, including cache-write rate.
                    estimated_cost_usd=(usage.input_tokens * input_rate + usage.output_tokens * output_rate) / 1_000_000,
                )
            telemetry = telemetry.model_copy(update=updates)
            if response.status != "completed":
                raise ProviderUnavailable("incomplete_response")
            if any(getattr(part, "type", "") == "refusal"
                   for item in response.output for part in getattr(item, "content", [])):
                raise ProviderUnavailable("model_refusal")
            try:
                guidance = validate_guidance(MentorGuidance.model_validate_json(response.output_text), context)
            except (ValueError, TypeError):
                raise ProviderUnavailable("validation_failed") from None
        except ProviderUnavailable as exc:
            error = exc.code
        except Exception as exc:
            # Match SDK types without ever serializing exception strings.
            from openai import APITimeoutError, RateLimitError, APIConnectionError, AuthenticationError, NotFoundError, BadRequestError
            error = ("timeout" if isinstance(exc, APITimeoutError) else
                     "rate_limit" if isinstance(exc, RateLimitError) else
                     "authentication_failed" if isinstance(exc, AuthenticationError) else
                     "model_unavailable" if isinstance(exc, NotFoundError) else
                     "request_rejected" if isinstance(exc, BadRequestError) else
                     "connection_error" if isinstance(exc, APIConnectionError) else "api_error")
        telemetry = telemetry.model_copy(update={
            "latency_ms": int((time.monotonic() - started) * 1000),
            "success": error is None, "failure_code": error,
        })
        try:
            self.ledger.finish(run_id, telemetry)
        except Exception:
            # The durable reservation remains even if detailed telemetry fails.
            raise ProviderUnavailable("telemetry_storage_error", telemetry) from None
        if error:
            raise ProviderUnavailable(error, telemetry) from None
        return ProviderResult(guidance=guidance, telemetry=telemetry)
