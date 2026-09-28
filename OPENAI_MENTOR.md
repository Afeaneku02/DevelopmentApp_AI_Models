# Local OpenAI mentor

The existing deterministic `GET /users/{user_id}/mentor-feedback` remains unchanged.
`POST /users/{user_id}/mentor-guidance` adds read-only, policy-filtered guidance through
the official Python SDK and Responses API. It never creates beliefs, evidence,
recommendation records, or roadmaps. A separate usage ledger is the only write made
by the provider. This is a local alpha, not an authenticated public service.

## Start and try it

Use the project's virtual environment on Windows:

```powershell
.\.venv\Scripts\python.exe tools/serve_api.py --db canonical.sqlite3
```

For a fresh installation, create a virtual environment and install `requirements.txt`.
Put your key in the ignored `.env` at the project root, never in `.env.example`.
The launcher explicitly reads this file; process environment variables take priority.
Tests and the app factory do not automatically load local secrets.

Use that same virtual-environment interpreter when installing dependencies:
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
A server launched with a different Python installation may lack the OpenAI SDK
even when it can run the API. Check `/health` for the database path and ensure
DevelopmentApp's `AI_MODELS_BASE_URL` points to that server. Preserve the database
containing the app's activity when restarting; a different database will not
contain the same user beliefs. The default local URL is `http://127.0.0.1:8100`.

Open `http://127.0.0.1:8100/docs` and select `POST /users/{user_id}/mentor-guidance`.
Use a user ID already present in your database and this request:

```json
{
  "context_key": "fitness_scheduling",
  "goal": "Build a consistent walking habit",
  "question": "What small step could I try this week?"
}
```

Optional `clarifications` carries a bounded, per-request clarification context: the
mentor's earlier clarifying questions in one interaction and the user's answers, oldest
first (at most 2 turns; `question` and `answer` each non-blank and at most 500 characters).
The client resends them each request; this service keeps no conversation state, never
stores them, and never turns them into evidence. They pass the same high-stakes filter as
`goal`/`question` and never bypass the risk-policy or evidence gates. Prompt version
`mentor-guidance-2` tells the model to read each answer as a reply to its question.

```json
{
  "context_key": "fitness_scheduling",
  "goal": "Get fitter",
  "clarifications": [
    {"question": "How much time could you realistically set aside each week?", "answer": "10 minutes"}
  ]
}
```

The current low-risk contexts are `fitness_scheduling` and `habit_nudge`. A user needs
authorized, sufficiently supported beliefs. Missing evidence or consequential contexts
return a clarification without an API call. Existing event intake and recomputation
remain responsible for building the user model; this endpoint does not run that pipeline.

Response status distinguishes `openai`, `mock`, `needs_more_information`,
`deterministic_fallback`, and `unavailable`. `reason_code` explains failures without
including SDK exception text. Generated actions are suggestions, never executions.
No web search is enabled; a `needs_web` response is reported as unavailable.

## Limits and telemetry

Defaults are a **$1 estimated experiment ceiling**, **100 attempted calls**, 1,200
maximum output tokens, a 20-second SDK timeout and **zero automatic retries**.
The smoke/evaluation tools further cap their experiment ceiling at $0.05. These all
share `.local/mentor-usage.sqlite3`, so calls remain counted across process restarts.
Unknown usage after a timeout keeps its reservation; all reservations are retained
even after success. The ceiling is intentionally more conservative than actual usage.

Optional server-side environment variables (not needed in `.env.example`):

| Variable | Default |
| --- | --- |
| `OPENAI_MODEL` | `gpt-5.6-luna` |
| `MENTOR_BUDGET_USD` | `1.0` (maximum 20) |
| `MENTOR_MAX_CALLS` | `100` |
| `MENTOR_MAX_OUTPUT_TOKENS` | `1200` |
| `MENTOR_TIMEOUT_SECONDS` | `20` |
| `MENTOR_LEDGER_PATH` | Project `.local/mentor-usage.sqlite3` |

Prices are versioned in `src/llm/budget.py` for Luna, Terra and Sol; an unpriced
model fails closed before a call. Estimates use standard short-context prices and
the cache-write input rate conservatively. Recheck prices before larger experiments.
This local estimate does not replace OpenAI billing or account spend limits. Deleting
the ledger, using another ledger path or changing configuration starts/changes its scope.

Telemetry includes the operation/request IDs, configured/returned model, prompt/price
versions, latency, token usage, estimated cost, reserved cost and success/failure.
Default telemetry contains no prompts, belief values, API keys or generated text.
SDK request-body debug logging is disabled. A snapshot hash and approved belief IDs
in the response support traceability without sending the full database to OpenAI.

## Test commands

```powershell
# No network or API credits:
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe tools/evaluate_user_model.py --manifest examples/evals
.\.venv\Scripts\python.exe tools/mentor_smoke.py
.\.venv\Scripts\python.exe tools/evaluate_mentor_guidance.py

# Explicit paid tests, fictional data only:
.\.venv\Scripts\python.exe tools/mentor_smoke.py --live
.\.venv\Scripts\python.exe tools/evaluate_mentor_guidance.py --live
```

The small evaluation covers ordinary habits, an instruction-injection request, missing
evidence, unknown context and high risk. Only three cases can call the provider.
It verifies policy behavior, response structure and unchanged database bytes, and saves
validated fictional guidance alongside deterministic baselines under `.local/` for
human review. Passing these checks does not prove semantic correctness or advice quality.

## Boundaries and remaining work

### Initial evaluation results (2026-09-20)

All 809 unittest tests, eight existing user-model evaluations, six mock mentor
scenarios, and six live mentor scenarios passed. The live batch made
three successful fictional-data calls, using 2,717 tokens with a conservative
usage estimate of $0.0013433. Three earlier sandbox connection failures exercised
deterministic fallback; their reservations remain counted. Including the earlier
smoke test, the ledger records seven attempts, $0.019318 reserved, and $0.0017314
estimated reported usage. These are local estimates, not a billing reconciliation.

Review of the three generated responses found small, reversible suggestions,
uncertainty language, and no claim that the injected instruction changed beliefs
or saved evidence. The model did narrow a general adherence belief to walking in
two explanations; this illustrates why valid belief IDs do not prove exact semantic
grounding. Broader grounding evaluation remains necessary.

Contract review: **Pass for this read-only milestone** against blueprint sections
6.4, 6.5, 18 and 22. Blueprint extraction reported no warnings. Backend policy owns
risk and eligibility; strict output validation rejects authority fields; regression
fixtures cover exclusions and database immutability. The contract validator passed
one representative belief and three authorized evidence records. The new guidance
envelope is not a persisted recommendation record.

The context builder always calls `authorize_beliefs_for_context`, then applies mentor
lifecycle/evidence checks and excludes session-only/do-not-persist beliefs. It sends
at most five approved belief summaries and provenance IDs. Raw events, observations,
reasoning text and outcome feedback are excluded. Higher-risk/manual-review contexts
cannot receive LLM-generated actions in this milestone.

Strict schemas and belief-ID checks reject malformed responses, invented references,
extra authority fields, refusals and incomplete generations. The existing high-stakes
wording filter supplements these checks; it is not an exhaustive semantic classifier.
Prompt instructions cannot make arbitrary model text perfectly grounded. Human review
of the fictional evaluation and broader adversarial tests remain necessary before
production use. No multi-turn conversation memory or automatic learning is implemented.

Authentication and per-user access control are still TODO in the existing API. The
launcher refuses a non-loopback bind when an API key is configured. Do not expose this
internal app through another server or tunnel before implementing authentication.

Official references: [Responses structured output](https://developers.openai.com/api/docs/guides/structured-outputs),
[Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[pricing](https://developers.openai.com/api/docs/pricing).
