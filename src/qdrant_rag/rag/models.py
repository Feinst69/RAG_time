"""Pydantic models for tickets, filters and search results."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# Ticket payload (mirrors the CSV schema stored inside Qdrant)
# ---------------------------------------------------------------------------

class TicketPayload(BaseModel):
    ticket_id: int
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    product: Optional[str] = None
    category: Optional[str] = None
    issue_description: Optional[str] = None
    resolution_notes: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    channel: Optional[str] = None
    region: Optional[str] = None
    customer_age: Optional[int] = None
    customer_gender: Optional[str] = None
    subscription_type: Optional[str] = None
    customer_tenure_months: Optional[int] = None
    previous_tickets: Optional[int] = None
    customer_satisfaction_score: Optional[float] = None
    first_response_time_hours: Optional[float] = None
    resolution_time_hours: Optional[float] = None
    ticket_created_date: Optional[str] = None   # ISO-8601 date string
    ticket_resolved_date: Optional[str] = None  # ISO-8601 date string
    escalated: Optional[bool] = None
    sla_breached: Optional[bool] = None
    operating_system: Optional[str] = None
    browser: Optional[str] = None
    payment_method: Optional[str] = None
    language: Optional[str] = None
    preferred_contact_time: Optional[str] = None
    issue_complexity_score: Optional[int] = None
    customer_segment: Optional[str] = None


# ---------------------------------------------------------------------------
# Filter parameters
# Keyword fields use exact-match.
# Numeric fields offer _min / _max bounds (either or both can be set).
# Date fields use ISO strings ("2023-01-01"); either bound is optional.
# Bool fields map to Python bool.
# ---------------------------------------------------------------------------

class FilterParams(BaseModel):
    # --- keyword (exact match) ---
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    product: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    channel: Optional[str] = None
    region: Optional[str] = None
    customer_gender: Optional[str] = None
    subscription_type: Optional[str] = None
    operating_system: Optional[str] = None
    browser: Optional[str] = None
    payment_method: Optional[str] = None
    language: Optional[str] = None
    customer_segment: Optional[str] = None

    # --- integer ranges ---
    customer_age_min: Optional[int] = Field(None, ge=0)
    customer_age_max: Optional[int] = Field(None, ge=0)
    issue_complexity_score_min: Optional[int] = Field(None, ge=0)
    issue_complexity_score_max: Optional[int] = Field(None, ge=0)

    # --- float ranges ---
    customer_satisfaction_score_min: Optional[float] = Field(None, ge=0)
    customer_satisfaction_score_max: Optional[float] = Field(None, ge=0)
    resolution_time_hours_min: Optional[float] = Field(None, ge=0)
    resolution_time_hours_max: Optional[float] = Field(None, ge=0)

    # --- datetime ranges (ISO-8601, e.g. "2023-06-01") ---
    ticket_created_date_from: Optional[str] = None
    ticket_created_date_to: Optional[str] = None
    ticket_resolved_date_from: Optional[str] = None
    ticket_resolved_date_to: Optional[str] = None

    # --- booleans ---
    escalated: Optional[bool] = None
    sla_breached: Optional[bool] = None

    def is_empty(self) -> bool:
        return all(v is None for v in self.model_dump().values())


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    ticket_id: int
    score: float
    payload: dict[str, Any]

    def __str__(self) -> str:
        lines = [
            f"[ticket_id={self.ticket_id}  score={self.score:.4f}]",
            f"  product    : {self.payload.get('product')}",
            f"  category   : {self.payload.get('category')}",
            f"  priority   : {self.payload.get('priority')}",
            f"  customer   : {self.payload.get('customer_name')}",
            f"  issue      : {str(self.payload.get('issue_description', ''))[:120]}",
            f"  resolution : {str(self.payload.get('resolution_notes', ''))[:120]}",
        ]
        return "\n".join(lines)
