#!/usr/bin/env python3
"""Helpers for recording scouting token and money cost."""

from __future__ import annotations


def zero_cost(provider: str = "openalex") -> dict:
    return {
        "provider": provider,
        "token_count": 0,
        "money_cost_usd": 0.0,
        "currency": "USD",
        "note": "OpenAlex discovery does not invoke a model; token usage is zero.",
    }
