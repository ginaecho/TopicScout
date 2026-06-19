import sys
import os
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_corpus import classify
from build_dashboard import graph_data, wiki_data
from costs import zero_cost
from init_topic import build_queries, parse_years, slugify
from intent_refiner import refine_intent, refine_intent_codex
from paper_graph import Candidate, relevance


CONFIG = {
    "topic": "AI theorem proving",
    "goal": "Track proof systems",
    "audience": "researchers",
    "include": ["proof search", "formal verification"],
    "exclude": ["math tutoring"],
    "years": {"from": 2023, "to": 2026},
    "evidence_types": ["methods", "benchmarks"],
    "taxonomy": ["proof search", "verification", "benchmarks"],
}


class CoreTests(unittest.TestCase):
    def test_topic_helpers(self):
        self.assertEqual(slugify("AI for Theorem Proving!"), "ai-for-theorem-proving")
        self.assertEqual(parse_years("2023-2026"), {"from": 2023, "to": 2026})
        self.assertIn("AI theorem proving proof search", build_queries(CONFIG["topic"], CONFIG["include"], CONFIG["evidence_types"]))

    def test_relevance_respects_include_and_exclude(self):
        candidate = Candidate(
            "openalex:1", "Formal proof search", 2025, "https://example.com", None,
            "Formal verification with proof search.", 10, [], ["query"]
        )
        score, reason = relevance(candidate, CONFIG)
        self.assertGreater(score, 0)
        self.assertIn("proof search", reason)

    def test_classification_graph_and_wiki(self):
        papers = [
            {
                "id": "openalex:a", "title": "Proof Search A", "abstract": "proof search benchmark",
                "topics": [], "primary_category": "unclassified", "url": "https://a", "year": 2025,
            },
            {
                "id": "openalex:b", "title": "Proof Search B", "abstract": "proof search method",
                "topics": [], "primary_category": "unclassified", "url": "https://b", "year": 2026,
            },
        ]
        for paper in papers:
            paper["primary_category"] = classify(paper, CONFIG)
        graph = graph_data(papers)
        wiki = wiki_data(CONFIG, papers, graph)
        self.assertEqual(len(graph["nodes"]), 2)
        self.assertGreaterEqual(wiki["page_count"], 6)
        self.assertIn("overview", {page["id"] for page in wiki["pages"]})

    def test_noninteractive_initialization_generates_agent_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, TOPIC_SCOUT_ROOT=directory)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "init_topic.py"),
                    "--topic", "AI theorem proving",
                    "--goal", "Track verified proof systems",
                    "--audience", "research engineers",
                    "--include", "proof search,formal verification",
                    "--exclude", "math tutoring",
                    "--years", "2023-2026",
                    "--taxonomy", "proof search,verification,benchmarks",
                    "--offline",
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            root = Path(directory)
            self.assertTrue((root / "topic.json").exists())
            self.assertTrue((root / "AGENTS.md").exists())
            self.assertTrue((root / "agents" / "graph-scout.md").exists())
            self.assertTrue((root / "skills" / "topic-paper-scout" / "SKILL.md").exists())
            topic = json.loads((root / "topic.json").read_text())
            self.assertEqual(topic["raw_intent"], "AI theorem proving")
            self.assertIn("dashboard_sections", topic)

    def test_intent_refinement_uses_structured_responses_output(self):
        refined = {
            "title": "Business-Aligned Evaluation of AI Agents",
            "research_question": "How should AI agent answers be evaluated against business, policy, and system requirements?",
            "goal": "Guide evaluation design",
            "audience": "researchers",
            "include": ["answer quality", "business utility"],
            "exclude": [],
            "evidence_types": ["methods", "benchmarks"],
            "taxonomy": ["metrics", "governance", "agent systems"],
            "dashboard_sections": ["scorecards", "policy coverage", "system design"],
            "search_queries": ["query one", "query two", "query three", "query four", "query five"],
            "scouting_strategy": ["Search evaluation benchmarks", "Expand citation graphs", "Test policy terminology"],
        }
        response_body = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(refined)}],
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return json.dumps(response_body).encode()

        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = timeout
            return FakeResponse()

        result, model = refine_intent(
            "raw sentence",
            provider="api",
            api_key="test-key",
            model="test-model",
            urlopen=fake_urlopen,
        )
        self.assertEqual(result["title"], refined["title"])
        self.assertEqual(model, "test-model")
        self.assertEqual(captured["payload"]["text"]["format"]["type"], "json_schema")
        self.assertFalse(captured["payload"]["store"])

    def test_cost_block_defaults_to_zero_for_openalex_scouting(self):
        cost = zero_cost()
        self.assertEqual(cost["token_count"], 0)
        self.assertEqual(cost["money_cost_usd"], 0.0)
        self.assertEqual(cost["currency"], "USD")

    def test_dashboard_payload_includes_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir(parents=True, exist_ok=True)
            (root / "reports").mkdir(parents=True, exist_ok=True)
            (root / "topic.json").write_text(
                json.dumps(
                    {
                        "topic": "Detecting Discriminatory AI in Hiring",
                        "goal": "Track bias detection methods",
                        "audience": "researchers",
                        "taxonomy": ["methods", "systems"],
                        "include": ["hiring", "fairness"],
                        "exclude": [],
                        "years": {"from": 2023, "to": 2026},
                    }
                ),
                encoding="utf-8",
            )
            (root / "data" / "papers.json").write_text(
                json.dumps({"papers": [], "scout_runs": []}),
                encoding="utf-8",
            )
            (root / "data" / "candidates.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-06-13T00:00:00Z",
                        "cost": {"token_count": 12, "money_cost_usd": 0.03, "currency": "USD"},
                        "queries": ["query"],
                        "candidates": [
                            {
                                "id": "openalex:1",
                                "title": "Candidate One",
                                "year": 2025,
                                "url": "https://example.com",
                                "citation_count": 4,
                                "relevance_score": 1.5,
                                "relevance_reason": "example",
                                "topics": ["fairness"],
                                "discovered_via": ["query:test"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            env = dict(os.environ, TOPIC_SCOUT_ROOT=directory)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "build_dashboard.py")],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            payload = json.loads((root / "data" / "dashboard.json").read_text())
            self.assertEqual(payload["candidate_count"], 1)
            self.assertEqual(payload["candidate_cost"]["token_count"], 12)
            self.assertEqual(payload["candidates"][0]["title"], "Candidate One")
            self.assertTrue(any(page["type"] == "candidate" for page in payload["wiki"]["pages"]))

    def test_dashboard_uses_candidates_for_visuals_when_no_papers_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir(parents=True, exist_ok=True)
            (root / "reports").mkdir(parents=True, exist_ok=True)
            (root / "topic.json").write_text(
                json.dumps(
                    {
                        "topic": "Detecting Discriminatory AI in Hiring",
                        "goal": "Track bias detection methods",
                        "audience": "researchers",
                        "taxonomy": ["methods", "systems"],
                        "include": ["hiring", "fairness"],
                        "exclude": [],
                        "years": {"from": 2023, "to": 2026},
                    }
                ),
                encoding="utf-8",
            )
            (root / "data" / "papers.json").write_text(
                json.dumps({"papers": [], "scout_runs": []}),
                encoding="utf-8",
            )
            (root / "data" / "candidates.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-06-13T00:00:00Z",
                        "cost": {"token_count": 12, "money_cost_usd": 0.03, "currency": "USD"},
                        "queries": ["query"],
                        "candidates": [
                            {
                                "id": "openalex:1",
                                "title": "Candidate One",
                                "year": 2025,
                                "url": "https://example.com",
                                "citation_count": 4,
                                "relevance_score": 1.5,
                                "relevance_reason": "example",
                                "topics": ["fairness"],
                                "discovered_via": ["query:test"],
                                "abstract": "fairness in hiring systems",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            env = dict(os.environ, TOPIC_SCOUT_ROOT=directory)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "build_dashboard.py")],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            payload = json.loads((root / "data" / "dashboard.json").read_text())
            self.assertEqual(payload["visualized_paper_count"], 1)
            self.assertEqual(payload["visualized_source"], "discovered candidates")
            self.assertGreater(len(payload["graph"]["nodes"]), 0)
            self.assertGreater(payload["categories"][0]["count"] + payload["categories"][1]["count"], 0)

    def test_dashboard_records_category_history_for_scout_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir(parents=True, exist_ok=True)
            (root / "reports").mkdir(parents=True, exist_ok=True)
            (root / "topic.json").write_text(
                json.dumps(
                    {
                        "topic": "Detecting Discriminatory AI in Hiring",
                        "goal": "Track bias detection methods",
                        "audience": "researchers",
                        "taxonomy": ["methods", "systems"],
                        "include": ["hiring", "fairness"],
                        "exclude": [],
                        "years": {"from": 2023, "to": 2026},
                    }
                ),
                encoding="utf-8",
            )
            (root / "data" / "papers.json").write_text(
                json.dumps(
                    {
                        "papers": [
                            {
                                "id": "openalex:1",
                                "title": "Candidate One",
                                "year": 2025,
                                "url": "https://example.com/1",
                                "citation_count": 4,
                                "abstract": "fairness in hiring systems",
                                "topics": ["fairness"],
                            },
                            {
                                "id": "openalex:2",
                                "title": "Candidate Two",
                                "year": 2025,
                                "url": "https://example.com/2",
                                "citation_count": 2,
                                "abstract": "hiring workflow systems",
                                "topics": ["workflow"],
                            },
                        ],
                        "scout_runs": [
                            {
                                "date": "2026-06-13",
                                "accepted_ids": ["openalex:1", "openalex:2"],
                                "accepted_count": 2,
                                "candidate_count": 10,
                                "cost": {"token_count": 0, "money_cost_usd": 0.0, "currency": "USD"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "data" / "candidates.json").write_text(
                json.dumps({"generated_at": "2026-06-13T00:00:00Z", "candidates": []}),
                encoding="utf-8",
            )
            env = dict(os.environ, TOPIC_SCOUT_ROOT=directory)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "build_dashboard.py")],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            payload = json.loads((root / "data" / "dashboard.json").read_text())
            self.assertEqual(len(payload["runs"]), 1)
            self.assertIn("accepted_topic_counts", payload["runs"][0])
            self.assertIn("cumulative_topics", payload["runs"][0])
            self.assertIn("cumulative_topic_ratios", payload["runs"][0])
            self.assertEqual(
                sum(payload["runs"][0]["accepted_topic_counts"].values()),
                payload["runs"][0]["accepted"],
            )
            self.assertEqual(
                payload["runs"][0]["cumulative_topics"],
                payload["runs"][0]["accepted_topic_counts"],
            )
            self.assertAlmostEqual(
                sum(payload["runs"][0]["cumulative_topic_ratios"].values()),
                1.0,
            )

    def test_scout_records_history_even_when_nothing_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, TOPIC_SCOUT_ROOT=directory)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "init_topic.py"),
                    "--topic", "AI theorem proving",
                    "--goal", "Track proof systems",
                    "--audience", "researchers",
                    "--include", "proof search,formal verification",
                    "--years", "2023-2026",
                    "--taxonomy", "proof search,verification,benchmarks",
                    "--offline",
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "scout.py")],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            corpus = json.loads((Path(directory) / "data" / "papers.json").read_text())
            self.assertEqual(len(corpus["scout_runs"]), 1)
            self.assertEqual(corpus["scout_runs"][0]["accepted_count"], 0)
            self.assertGreaterEqual(corpus["scout_runs"][0]["candidate_count"], 0)

    def test_intent_refinement_can_use_codex_subscription(self):
        refined = {
            "title": "Business-Aligned Evaluation of AI Agents",
            "research_question": "How should AI agent answers be evaluated against business, policy, and system requirements?",
            "goal": "Guide evaluation design",
            "audience": "researchers",
            "include": ["answer quality", "business utility"],
            "exclude": [],
            "evidence_types": ["methods", "benchmarks"],
            "taxonomy": ["metrics", "governance", "agent systems"],
            "dashboard_sections": ["scorecards", "policy coverage", "system design"],
            "search_queries": ["query one", "query two", "query three", "query four", "query five"],
            "scouting_strategy": ["Search evaluation benchmarks", "Expand citation graphs", "Test policy terminology"],
        }
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text(json.dumps(refined), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        result, model = refine_intent_codex(
            "raw sentence",
            cwd=ROOT,
            run=fake_run,
        )
        self.assertEqual(result["title"], refined["title"])
        self.assertEqual(model, "codex-cli:configured-model")
        self.assertIn("--ephemeral", captured["command"])
        self.assertIn("--ignore-user-config", captured["command"])
        self.assertIn("--ignore-rules", captured["command"])
        self.assertIn("read-only", captured["command"])
        self.assertIn("--output-schema", captured["command"])


if __name__ == "__main__":
    unittest.main()
