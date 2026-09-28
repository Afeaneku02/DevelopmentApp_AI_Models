# OpenAI mentor integration preparation

Reviewed 2026-09-20. This document records the original preparation plan. The first
integration is now implemented; see [OPENAI_MENTOR.md](OPENAI_MENTOR.md) for current
behavior and usage. No paid API calls were made during the original preparation pass.

Baseline verification: all 787 unittest tests passed; all 8 evaluation scenarios passed. Git ignore checks confirmed `.env` and `.env.local` are ignored and `.env.example` is not. Existing untracked `canonical.sqlite3` and `demo_view.html` were left unchanged.

## Current architecture

- `src/orchestration/process_user_pipeline.py` coordinates events -> observations -> evidence -> recomputation, with user isolation and transactional writes.
- `src/recommendations/context_policy.py` owns `authorize_beliefs_for_context`, lifecycle exclusions, sensitivity restrictions, and context/risk resolution.
- `src/mentor/feedback.py` generates deterministic feedback from beliefs and active evidence. It remains the baseline and fallback.
- `src/api/app.py` exposes `GET /users/{user_id}/mentor-feedback` using a read-only repository. Preserve its contract.
- `src/storage/repository.py` owns persistence. The new provider must never receive this repository or any write capabilities.
- `requirements.txt` has Pydantic, FastAPI and HTTPX, but no OpenAI SDK. No provider boundary exists yet.
- The API is local internal alpha: `require_alpha_access` is a no-op. Real authentication and user authorization are prerequisites for external use, including any paid endpoint.

Two integration details matter: existing mentor feedback permits missing context and then skips context authorization; the new OpenAI path must require a nonblank context and always authorize. Existing goal fields are accepted but do not influence deterministic feedback yet.

The repository's pre-implementation-check and apply-risk-policy skills are explicitly placeholders. Before implementation, read relevant sections of blueprint v0.6.2 using the shared document reader, inspect its warnings, and run the contract-review workflow after the change. This preparation pass inspected source and tests, not the binary blueprint.

## Proposed first milestone

Add a separate `POST /users/{user_id}/mentor-guidance` endpoint, backed by a read-only repository even though the HTTP method is POST. POST makes the potentially billable generation explicit. Require context; accept a bounded goal and optional user question. Return an envelope distinguishing `openai`, `deterministic_fallback`, `needs_more_information`, and `unavailable`, with a non-secret reason code.

Create `src/llm/` with provider interface, strict input/output models, OpenAI and mock providers, prompts, and validation. Add a small mentor service/context builder under `src/mentor/`. Only the OpenAI provider imports/calls the official SDK. Read model selection once through configuration, defaulting to `gpt-5.6-luna`.

Context assembly must scope reads to the user, require policy authorization, check active supporting evidence and lifecycle eligibility, and include only bounded, relevant approved belief fields. Preserve belief IDs, confidence, status and policy/snapshot provenance. Start without raw event text or outcome feedback: authorization of a belief does not authorize every underlying free-text record for export. Future event/outcome inclusion needs its own minimization rules.

Use the Responses API with structured output. Proposed guidance fields: `summary`, `recommendations` (action, reason, grounded belief IDs), nullable `clarifying_question`, `needs_more_information`, and `needs_web`. No tools, web access, autonomous actions, memory ownership, evidence creation, belief writes, or roadmap changes. Treat user-supplied content as data, not instructions.

Backend validation must reject extra fields, malformed output, unknown or unauthorized IDs, inconsistent clarification flags, and recommendations outside backend policy. Require grounded references for personalized actions. Valid IDs alone do not prove the text follows from those beliefs: evaluate semantic grounding separately. For the first milestone, restrict actionable LLM guidance to approved low-risk contexts; route consequential/uncertain cases to clarification or existing deterministic behavior. The model cannot lower risk or waive confirmation/manual review.

Handle missing keys, timeouts, rate limits, API errors, refusals, incomplete responses and validation failures with an explicit fallback/unavailable result. Never return raw SDK exceptions, raw rejected output, or personal context in logs. All paths must leave user state unchanged.

## Configuration and key handling

`.env.example` contains only the empty API key and default model. `.gitignore` excludes `.env` and local variants while allowing the example. The current app does not load `.env`; implementation must either explicitly add local dotenv loading or document server process environment setup.

Create a dedicated development API project and supply `OPENAI_API_KEY` locally to the server when live testing begins. Do not paste the key into chat, source, frontend, tests or documentation. Confirm project model access with one bounded smoke call. Account credit and model access have not been verified in this review.

## Testing with approximately $20

Use mock calls for all automated tests. Suggested allocation: $1 for smoke tests, $4 for a small synthetic-user evaluation, $5 for prompt iteration, and $10 held back. These are experiment budgets, not provider-enforced caps.

Official Luna standard short-context rates checked on the review date: $0.20 per million input tokens and $1.20 per million output tokens. A hypothetical 3,000-input/1,000-billable-output call costs about $0.0018, or $0.18 per 100 calls, before cache-write charges and other applicable differences. Reasoning consumption, retries and larger outputs can raise spend; this is not a guaranteed call allowance.

Before paid batches, implement bounded context and output, explicit timeouts, bounded retries, a call-count limit and a local spend ledger. Reserve a conservative maximum cost before each request, count retries, and stop when the next reservation exceeds the experiment budget. Unknown usage after a timeout must retain a conservative reservation. Start sequentially, with 10 smoke calls, then 100 evaluation calls. Reconcile recorded usage with the API dashboard. Do not rely solely on dashboard alerts as a hard stop.

Record provider, configured/returned model, request ID, latency, token usage (including available cached/reasoning detail), outcome, safe error category and price-versioned estimated cost. No prompts, personal context, keys or raw responses in default telemetry. Preserve a synthetic scenario/run ID for later BetterYou_Simulator cost-per-user/day comparisons; simulator integration is not implemented here.

## Acceptance checks for implementation

1. Key configuration stays server-side; tracked files and logs contain no credentials.
2. Provider is injectable and all automated tests work without credentials/network.
3. Unauthorized, sensitive, restricted, locked, contested and wrong-user beliefs cannot enter the payload.
4. Missing/unknown context fails closed; the existing endpoint remains compatible.
5. Provider has no persistence capability; success and every failure path leave the database unchanged.
6. Malformed/extra-field/refusal/incomplete responses and invented belief IDs are rejected.
7. Valid structured output and clarification responses pass; inconsistent flags fail.
8. Backend risk/confirmation decisions constrain actions independently of model output.
9. Missing key, timeout, rate limit and API failure produce explicit fallback/unavailable states.
10. Model override, bounded output, retries, usage telemetry and budget stops are tested.
11. Prompt-injection fixtures and semantic-grounding evaluations supplement schema checks.
12. Existing unittest suite and evaluation scenarios continue passing; complete contract review.

## Official references

- [GPT-5.6 Luna model, supported features and rates](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- [API pricing](https://developers.openai.com/api/docs/pricing)
- [Structured model outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
