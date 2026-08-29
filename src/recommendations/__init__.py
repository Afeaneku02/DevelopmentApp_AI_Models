"""Recommendation-layer backend policy (blueprint section 6.4-6.5).

This phase implements only the *context / risk policy foundation*: resolving
a (possibly LLM-labelled) recommendation ``context_key`` to a backend-owned
risk tier and belief-eligibility policy, and filtering a set of beliefs down
to the ones actually usable as recommendation inputs in that context. No
candidate ranking, exploration, or recommendation issuance lives here yet.
"""
