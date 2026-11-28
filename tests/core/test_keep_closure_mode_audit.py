# Author: lee-leonardo (NetID: ll37)
# Description: Independent synthetic expectations for graph closure analysis.
"""Graph closure analysis: mapping, provenance, and target-set monotonicity."""

import json
import tempfile
import unittest
from pathlib import Path

from examples.keep_closure_mode_audit import (
    ROOT,
    reference_graph,
    run_audit,
    structural_gaps,
    write_synthetic_bundle,
)
from pyhealth.graph import KnowledgeGraph
from pyhealth.medcode import OMOPVocabulary
from pyhealth.processors import AncestorRollupProcessor, UnmappedCodeError
from pyhealth.tasks import build_keep_omop_ancestor_rollup


class TestClosureAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "input"
        self.destination = Path(self.temp.name) / "results"
        write_synthetic_bundle(self.folder)
        self.vocabulary = OMOPVocabulary(self.folder)

    def build(self, mode="direct", policy="first_retained", **options):
        return build_keep_omop_ancestor_rollup(
            self.vocabulary,
            path_source=mode,
            target_policy=policy,
            include_root=False,
            **{"max_depth": 5, "on_unmapped": "empty", **options},
        )

    def test_exact_graph_sets_and_authoritative_depths(self):
        depths, direct, pairs = reference_graph(self.vocabulary)
        expected = {
            (ROOT, "100"),
            ("100", "101"),
            ("101", "102"),
            ("102", "103"),
            ("103", "104"),
            ("104", "200"),
            ("103", "204"),
            ("104", "204"),
            ("204", "205"),
        }
        self.assertEqual(direct, expected)
        self.assertEqual(pairs, expected | {("104", "201"), ("102", "204")})
        self.assertEqual(len(depths), 12)
        self.assertEqual(depths["204"], 6)  # Direct shortest path is five.
        for mode in ("direct", "closure"):
            processor = self.build(mode)
            self.assertNotIn("204", processor.retained_codes)
            self.assertEqual(processor.rollup_codes(["204"])[0].source_depth, 6)
            self.assertEqual(
                processor.retained_codes,
                frozenset(["100", "101", "102", "103", "104", "203"]),
            )

    def test_recovery_and_branch_local_added_target(self):
        direct, closure = self.build(), self.build("closure")
        self.assertEqual(direct.rollup_codes(["201"])[0].reason, "no_target")
        self.assertEqual(closure.rollup_codes(["201"])[0].targets, ("104",))
        self.assertEqual(direct.rollup_codes(["204"])[0].targets, ("103", "104"))
        self.assertEqual(
            closure.rollup_codes(["204"])[0].targets, ("102", "103", "104")
        )
        all_direct = self.build(policy="all_retained")
        all_closure = self.build("closure", "all_retained")
        expected = ("100", "101", "102", "103", "104")
        self.assertEqual(all_direct.rollup_codes(["204"])[0].targets, expected)
        self.assertEqual(all_closure.rollup_codes(["204"])[0].targets, expected)

    def test_filtered_intermediate_and_default_error(self):
        for mode in ("direct", "closure"):
            processor = self.build(mode)
            self.assertEqual(processor.rollup_codes(["202"])[0].reason, "no_target")
            self.assertEqual(processor.rollup_codes(["203"])[0].targets, ("203",))
            eligible = self.build(mode, max_depth=7)
            self.assertEqual(eligible.rollup_codes(["202"])[0].targets, ("202",))
        # Omit on_unmapped: exercise the actual production default.
        processor = build_keep_omop_ancestor_rollup(
            self.vocabulary, target_policy="first_retained", include_root=False
        )
        with self.assertRaises(UnmappedCodeError):
            processor.rollup_codes(["202"])

    def test_external_classification_is_aligned(self):
        for mode in ("direct", "closure"):
            results = self.build(mode).rollup_codes(["900", "901", "902", "903"])
            self.assertEqual(
                [r.reason for r in results],
                ["outside_domain", "unreachable", "unknown_code", "unknown_code"],
            )

    def test_provenance_and_projection_decode(self):
        processor = self.build("closure")
        self.assertEqual(processor.provenance["closure_only_edge_count"], "2")
        try:
            graph = processor.process(["204"])
        except ImportError:
            self.skipTest("torch-geometric not installed")
        codes = [processor.id2entity[node] for node in graph.node_ids.tolist()]
        edges = list(zip(*graph.edge_index.tolist()))
        decoded = [(codes[a], codes[b]) for a, b in edges]
        self.assertEqual(set(decoded), {("102", "204"), ("103", "204"), ("104", "204")})
        self.assertEqual(
            {edge for edge, flag in zip(decoded, graph.closure_only) if flag},
            {("102", "204")},
        )
        self.assertEqual(
            {codes[i] for i, flag in enumerate(graph.target_mask) if flag},
            {"102", "103", "104"},
        )

    def test_aggregates_reconcile_to_hand_counts(self):
        summary = run_audit(self.folder, self.destination, synthetic=True)
        self.assertEqual(
            summary["graph_counts"],
            {
                "U": 12,
                "D": 9,
                "P": 11,
                "intersection": 9,
                "P_minus_D": 2,
                "D_minus_P": 0,
                "union": 11,
            },
        )
        for policy, changed, identical in [
            ("first_retained", 2, 9),
            ("all_retained", 0, 11),
        ]:
            result = summary["policies"][policy]
            stats = result["populations"]["all"]
            self.assertEqual(stats["denominator"], 12)
            self.assertEqual(
                stats["direct_reasons"], {"retained": 6, "rolled_up": 3, "no_target": 3}
            )
            self.assertEqual(
                stats["closure_reasons"],
                {"retained": 6, "rolled_up": 4, "no_target": 2},
            )
            self.assertEqual(stats["categories"]["recovered_no_target"], 1)
            self.assertEqual(stats["categories"]["changed_nonempty_targets"], changed)
            self.assertEqual(stats["categories"]["identical_targets"], identical)
            self.assertEqual(result["populations"]["deep"]["denominator"], 5)
            self.assertEqual(result["populations"]["deep"]["direct_no_target"], 2)
            self.assertEqual(result["populations"]["deep"]["closure_no_target"], 1)
            self.assertEqual(
                result["structural_gaps"]["direct"]["root_unreachable"],
                {"total": 3, "eligible": 1, "deep": 2, "excluded_root": 0},
            )
            self.assertEqual(
                result["structural_gaps"]["closure"]["root_unreachable"],
                {"total": 2, "eligible": 1, "deep": 1, "excluded_root": 0},
            )
            lines = (
                (self.destination / f"{policy}_differences.tsv")
                .read_text()
                .splitlines()
            )
            self.assertEqual(len(lines), changed + 2)
        self.assertEqual(
            json.loads((self.destination / "summary.json").read_text())["graph_counts"],
            summary["graph_counts"],
        )

    def test_overlap_and_duplicate_order_invariance(self):
        write_synthetic_bundle(self.folder, overlay=False)
        vocabulary = OMOPVocabulary(self.folder)
        depths, direct, pairs = reference_graph(vocabulary)
        self.assertEqual(direct, pairs)
        for policy in ("first_retained", "all_retained"):
            config = {
                "target_policy": policy,
                "include_root": False,
                "on_unmapped": "empty",
            }
            a = build_keep_omop_ancestor_rollup(
                vocabulary, path_source="direct", **config
            )
            b = build_keep_omop_ancestor_rollup(
                vocabulary, path_source="closure", **config
            )
            self.assertEqual(
                a.rollup_codes(sorted(depths)), b.rollup_codes(sorted(depths))
            )
            self.assertEqual(b.provenance["closure_only_edge_count"], "0")
        write_synthetic_bundle(self.folder)
        before = self.build("closure")
        expected = reference_graph(self.vocabulary)
        write_synthetic_bundle(self.folder, duplicate_pairs=True)
        after = self.build("closure")
        self.assertEqual(reference_graph(self.vocabulary), expected)
        self.assertEqual(
            before.rollup_codes(sorted(depths)), after.rollup_codes(sorted(depths))
        )
        self.assertEqual(before.fingerprint, after.fingerprint)

    def test_both_policies_preserve_targets_when_edges_are_added(self):
        # Exhaust every possible acyclic overlay on this ordered four-node DAG.
        # Fixed depths/eligibility: paths to old first-retained targets survive.
        candidates = [
            ("R", "A"),
            ("R", "B"),
            ("R", "X"),
            ("A", "B"),
            ("A", "X"),
            ("B", "X"),
        ]
        graph = KnowledgeGraph(
            triples=[("A", "is_a", "X")],
            entity2id={c: i for i, c in enumerate(["R", "A", "B", "X"])},
            relation2id={"is_a": 0},
        )
        for policy in ("first_retained", "all_retained"):
            config = {
                "knowledge_graph": graph,
                "domain_by_code": {c: "Condition" for c in "RABX"},
                "hierarchy_relation": "is_a",
                "root": "R",
                "target_policy": policy,
                "include_root": False,
                "max_depth": 2,
                "on_unmapped": "empty",
                "root_depths": {"R": 0, "A": 1, "B": 2, "X": 3},
            }
            baseline = AncestorRollupProcessor(**config).rollup_codes(["X"])[0]
            for mask in range(1 << len(candidates)):
                pairs = [edge for i, edge in enumerate(candidates) if mask & (1 << i)]
                result = AncestorRollupProcessor(
                    **config, path_source="closure", closure_pairs=pairs
                ).rollup_codes(["X"])[0]
                self.assertLessEqual(set(baseline.targets), set(result.targets))

    def test_self_edges_and_cycles_rejected(self):
        graph = KnowledgeGraph(triples=[("R", "is_a", "A")])
        config = {
            "knowledge_graph": graph,
            "domain_by_code": {c: "Condition" for c in "RA"},
            "hierarchy_relation": "is_a",
            "root": "R",
            "target_policy": "first_retained",
            "include_root": False,
            "path_source": "closure",
        }
        for pairs in [[("A", "A")], [("A", "R")]]:
            with self.assertRaises(ValueError):
                AncestorRollupProcessor(**config, closure_pairs=pairs)

    def test_structural_deep_population_uses_cutoff_even_without_eligible_nodes(self):
        self.assertEqual(
            structural_gaps({ROOT: 0, "X": 2}, set(), set(), 5)["root_unreachable"],
            {"total": 1, "eligible": 0, "deep": 0, "excluded_root": 0},
        )
