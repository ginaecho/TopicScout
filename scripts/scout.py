#!/usr/bin/env python3
"""Discover candidate papers without silently accepting them."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from costs import zero_cost
from paper_graph import discover
from workspace import CANDIDATES_PATH, PAPERS_PATH, load_json, load_topic, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    parser.add_argument("--seed-limit", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument(
        "--accept-score",
        type=float,
        default=None,
        help="Accept candidates at or above this score when topic.json allows auto-approval",
    )
    args = parser.parse_args()
    config = load_topic()
    result = discover(config, args.query, args.seed_limit, args.neighbors)
    existing = load_json(PAPERS_PATH, {"papers": []})
    known = {paper["id"] for paper in existing.get("papers", [])}
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["cost"] = zero_cost()
    result["new_candidate_count"] = sum(
        candidate["id"] not in known for candidate in result["candidates"]
    )
    write_json(CANDIDATES_PATH, result)

    accepted = []
    scout_run = {
        "date": result["generated_at"][:10],
        "queries": result["queries"],
        "accepted_ids": [],
        "accepted_count": 0,
        "candidate_count": len(result["candidates"]),
        "cost": result["cost"],
    }
    existing.setdefault("scout_runs", []).append(scout_run)
    if not config["approval_required"] and args.accept_score is not None:
        accepted = [
            candidate for candidate in result["candidates"]
            if candidate["id"] not in known
            and candidate["relevance_score"] >= args.accept_score
            and candidate["abstract"]
        ]
        if accepted:
            for candidate in accepted:
                candidate.update(
                    {
                        "accepted_at": result["generated_at"],
                        "primary_category": "unclassified",
                        "categories": [],
                        "notes": "",
                    }
                )
            existing["papers"].extend(accepted)
            scout_run["accepted_ids"] = [paper["id"] for paper in accepted]
            scout_run["accepted_count"] = len(accepted)
            write_json(PAPERS_PATH, existing)
    else:
        write_json(PAPERS_PATH, existing)

    print(
        f"Discovered {len(result['candidates'])} candidates "
        f"({result['new_candidate_count']} not in corpus)."
    )
    if config["approval_required"]:
        print(f"Review {CANDIDATES_PATH}; acceptance requires a librarian or human.")
    elif accepted:
        print(f"Accepted {len(accepted)} candidates at score >= {args.accept_score}.")
    print(
        f"Scout cost: {result['cost']['token_count']} tokens, "
        f"${result['cost']['money_cost_usd']:.2f} {result['cost']['currency']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
