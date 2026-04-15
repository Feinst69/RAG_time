"""Convert FilterParams into a Qdrant Filter object.

Each field type is handled differently:
  - Keyword  → FieldCondition + MatchValue (exact string)
  - Integer  → FieldCondition + Range (gte / lte)
  - Float    → FieldCondition + Range (gte / lte)
  - Datetime → FieldCondition + DatetimeRange (gte / lte as ISO strings)
  - Bool     → FieldCondition + MatchValue (bool)

All active conditions are combined with AND semantics (must=[…]).
Returns None when no filter parameters are set.
"""

from __future__ import annotations

from typing import Optional

from qdrant_client.http import models as qm

from qdrant_rag.rag.models import FilterParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kw(field: str, value: str) -> qm.FieldCondition:
    return qm.FieldCondition(key=field, match=qm.MatchValue(value=value))


def _range(field: str, gte=None, lte=None) -> qm.FieldCondition:
    return qm.FieldCondition(key=field, range=qm.Range(gte=gte, lte=lte))


def _dt_range(field: str, gte: Optional[str] = None, lte: Optional[str] = None) -> qm.FieldCondition:
    """DatetimeRange expects RFC-3339 / ISO-8601 strings, e.g. '2023-06-01'."""
    return qm.FieldCondition(
        key=field,
        range=qm.DatetimeRange(
            gte=gte,
            lte=lte,
        ),
    )


def _bool(field: str, value: bool) -> qm.FieldCondition:
    return qm.FieldCondition(key=field, match=qm.MatchValue(value=value))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_filter(params: FilterParams) -> Optional[qm.Filter]:
    """Build a Qdrant ``Filter`` from *params*.

    Returns ``None`` when *params* carries no constraints so callers can pass
    it directly to ``query_points(filter=…)`` without special-casing.
    """
    if params.is_empty():
        return None

    must: list[qm.Condition] = []

    # ------------------------------------------------------------------ #
    # Keyword exact-match fields
    # ------------------------------------------------------------------ #
    keyword_map = {
        "customer_name": params.customer_name,
        "customer_email": params.customer_email,
        "product": params.product,
        "category": params.category,
        "priority": params.priority,
        "channel": params.channel,
        "region": params.region,
        "customer_gender": params.customer_gender,
        "subscription_type": params.subscription_type,
        "operating_system": params.operating_system,
        "browser": params.browser,
        "payment_method": params.payment_method,
        "language": params.language,
        "customer_segment": params.customer_segment,
    }
    for field, value in keyword_map.items():
        if value is not None:
            must.append(_kw(field, value))

    # ------------------------------------------------------------------ #
    # Integer range fields
    # ------------------------------------------------------------------ #
    if params.customer_age_min is not None or params.customer_age_max is not None:
        must.append(_range("customer_age", gte=params.customer_age_min, lte=params.customer_age_max))

    if params.issue_complexity_score_min is not None or params.issue_complexity_score_max is not None:
        must.append(_range(
            "issue_complexity_score",
            gte=params.issue_complexity_score_min,
            lte=params.issue_complexity_score_max,
        ))

    # ------------------------------------------------------------------ #
    # Float range fields
    # ------------------------------------------------------------------ #
    if params.customer_satisfaction_score_min is not None or params.customer_satisfaction_score_max is not None:
        must.append(_range(
            "customer_satisfaction_score",
            gte=params.customer_satisfaction_score_min,
            lte=params.customer_satisfaction_score_max,
        ))

    if params.resolution_time_hours_min is not None or params.resolution_time_hours_max is not None:
        must.append(_range(
            "resolution_time_hours",
            gte=params.resolution_time_hours_min,
            lte=params.resolution_time_hours_max,
        ))

    # ------------------------------------------------------------------ #
    # Datetime range fields (ISO-8601 strings)
    # ------------------------------------------------------------------ #
    if params.ticket_created_date_from is not None or params.ticket_created_date_to is not None:
        must.append(_dt_range(
            "ticket_created_date",
            gte=params.ticket_created_date_from,
            lte=params.ticket_created_date_to,
        ))

    if params.ticket_resolved_date_from is not None or params.ticket_resolved_date_to is not None:
        must.append(_dt_range(
            "ticket_resolved_date",
            gte=params.ticket_resolved_date_from,
            lte=params.ticket_resolved_date_to,
        ))

    # ------------------------------------------------------------------ #
    # Boolean fields
    # ------------------------------------------------------------------ #
    if params.escalated is not None:
        must.append(_bool("escalated", params.escalated))

    if params.sla_breached is not None:
        must.append(_bool("sla_breached", params.sla_breached))

    if not must:
        return None

    return qm.Filter(must=must)
