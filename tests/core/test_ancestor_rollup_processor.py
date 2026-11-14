"""Tests for pyhealth.processors.AncestorRollupProcessor (generic path).

Ports the focused fixtures of the standalone behavioural reference (the
"ancestor rollup" toy rehearsal, C-0009): a multi-parent diamond with a
root shortcut, canary rows of a non-hierarchy relation, supplied depths
that disagree with graph distances, both target policies, root inclusion,
duplicate occurrences, invalid DAGs, and exact projected edges and masks.

Mapping behaviour is tested without torch-geometric, following PyHealth's
optional-dependency convention; projection, batching, and disk round-trip
tests are gated on the ``graph`` extra.

Run: python -m pytest tests/core/test_ancestor_rollup_processor.py -v
"""

import pickle
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import torch

from pyhealth.graph import KnowledgeGraph
from pyhealth.processors import (
    AncestorRollupProcessor,
    RollupResult,
    UnmappedCodeError,
)


def _has_pyg():
    try:
        import torch_geometric  # noqa: F401

        return True
    except ImportError:
        return False


# The diamond mirrors the behavioural reference: X has two parents C and D
# (the multi-parent probe), U has a root shortcut (R -> U) alongside a
# longer path (R -> B -> D -> U), N is a non-Condition node reachable from
# the root, O is an isolated Drug node, W is a Condition entity that no
# is_a edge touches, and the C -> X edge row is duplicated to test
# deduplication. The "treats" rows are canaries: if the processor ever
# selected them as hierarchy edges, X would gain parent B and W would
# become reachable.
DIAMOND_KG_TRIPLES = [
    ("R", "is_a", "A"), ("R", "is_a", "B"), ("A", "is_a", "C"),
    ("B", "is_a", "D"), ("C", "is_a", "X"), ("D", "is_a", "X"),
    ("C", "is_a", "X"), ("X", "is_a", "Y"), ("R", "is_a", "U"),
    ("D", "is_a", "U"), ("R", "is_a", "N"),
    ("B", "treats", "X"),
    ("O", "treats", "W"),
]

DIAMOND_KG_DOMAINS = {
    "R": "Condition", "A": "Condition", "B": "Condition", "C": "Condition",
    "D": "Condition", "X": "Condition", "Y": "Condition", "U": "Condition",
    "N": "Observation", "O": "Drug", "W": "Condition",
}

#: Supplied depths that deliberately disagree with the graph distances: U is
#: at BFS depth 1 (root shortcut) but supplied as 3, so it rolls up instead
#: of staying a retained identity.
DISAGREEING_DEPTHS = {
    "R": 0, "A": 1, "B": 1, "C": 2, "D": 2, "X": 3, "Y": 4, "U": 3, "N": 1,
}


def build_processor(triples=None, domains=None, **overrides):
    """Builds a diamond processor with the reference's test-friendly defaults."""
    values = {
        "knowledge_graph": KnowledgeGraph(
            triples=DIAMOND_KG_TRIPLES if triples is None else triples
        ),
        "domain_by_code": DIAMOND_KG_DOMAINS if domains is None else domains,
        "hierarchy_relation": "is_a",
        "root": "R",
        "target_policy": "first_retained",
        "include_root": False,
        "max_depth": 2,
        "on_unmapped": "empty",
    }
    values.update(overrides)
    return values, AncestorRollupProcessor(**values)


def node_codes(processor, data):
    """Decodes a projected graph's node_ids back to concept codes."""
    return tuple(processor.id2entity[int(i)] for i in data.node_ids.tolist())


def edge_pairs(processor, data):
    """Decodes a projected graph's edges to (parent, child) code pairs."""
    codes = node_codes(processor, data)
    return tuple(
        (codes[int(data.edge_index[0][j])], codes[int(data.edge_index[1][j])])
        for j in range(data.num_edges)
    )


class TestConstruction(unittest.TestCase):
    """Construction, provenance, fingerprint, and validation."""

    def test_documented_defaults(self):
        processor = AncestorRollupProcessor(
            knowledge_graph=KnowledgeGraph(triples=DIAMOND_KG_TRIPLES),
            domain_by_code=DIAMOND_KG_DOMAINS,
            hierarchy_relation="is_a",
            root="R",
            target_policy="first_retained",
            include_root=False,
        )
        self.assertEqual(processor._config.max_depth, 5)
        self.assertEqual(processor._config.on_unmapped, "error")
        self.assertEqual(processor._config.path_source, "direct")

    def test_processor_hooks(self):
        _, processor = build_processor()
        processor.fit([], "field")  # no-op, does not raise
        self.assertFalse(processor.is_token())
        self.assertEqual(processor.schema(), ("graph",))
        self.assertEqual(processor.dim(), (0,))
        self.assertEqual(processor.spatial(), (False,))

    def test_provenance_records_the_depth_source_and_relation(self):
        _, processor = build_processor()
        provenance = dict(processor.provenance)
        self.assertEqual(provenance["depth_method"], "bfs_edge_distance")
        self.assertEqual(provenance["path_source"], "direct")
        self.assertEqual(provenance["hierarchy_relation"], "is_a")
        self.assertEqual(provenance["edge_orientation"], "parent_to_child")
        supplied = build_processor(root_depths=DISAGREEING_DEPTHS)[1]
        self.assertEqual(
            supplied.provenance["depth_method"], "supplied_root_depths"
        )

    def test_fingerprint_stable_for_identical_inputs(self):
        _, one = build_processor()
        _, two = build_processor()
        self.assertEqual(one.fingerprint, two.fingerprint)

    def test_fingerprint_sensitive_to_configuration(self):
        _, base = build_processor()
        _, other = build_processor(max_depth=3)
        self.assertNotEqual(base.fingerprint, other.fingerprint)
        _, other = build_processor(target_policy="all_retained")
        self.assertNotEqual(base.fingerprint, other.fingerprint)
        _, other = build_processor(include_root=True)
        self.assertNotEqual(base.fingerprint, other.fingerprint)
        _, other = build_processor(on_unmapped="error")
        self.assertNotEqual(base.fingerprint, other.fingerprint)

    def test_fingerprint_sensitive_to_hierarchy_and_domains(self):
        _, base = build_processor()
        triples = list(DIAMOND_KG_TRIPLES) + [("A", "is_a", "Y")]
        _, other = build_processor(triples=triples)
        self.assertNotEqual(base.fingerprint, other.fingerprint)
        domains = dict(DIAMOND_KG_DOMAINS, N="Procedure")
        _, other = build_processor(domains=domains)
        self.assertNotEqual(base.fingerprint, other.fingerprint)

    def test_fingerprint_sensitive_to_depth_source(self):
        _, base = build_processor()
        _, supplied = build_processor(root_depths=DISAGREEING_DEPTHS)
        self.assertNotEqual(base.fingerprint, supplied.fingerprint)

    def test_fingerprint_insensitive_to_unused_domain_entries(self):
        _, base = build_processor()
        domains = dict(DIAMOND_KG_DOMAINS, EXTRA="Condition")
        _, other = build_processor(domains=domains)
        self.assertEqual(base.fingerprint, other.fingerprint)

    def test_repr_shows_configuration_and_fingerprint(self):
        _, processor = build_processor()
        text = repr(processor)
        for expected in (
            "AncestorRollupProcessor", "is_a", "R", "first_retained",
            "max_depth=2", "path_source='direct'", "fingerprint=",
        ):
            self.assertIn(expected, text)

    def test_round_trip_preserves_fingerprint_and_mapping(self):
        _, processor = build_processor()
        clone = pickle.loads(pickle.dumps(processor))
        self.assertEqual(clone.fingerprint, processor.fingerprint)
        self.assertEqual(
            clone.rollup_codes(["X", "C"]), processor.rollup_codes(["X", "C"])
        )

    def test_knowledge_graph_missing_attributes(self):
        with self.assertRaises(TypeError):
            AncestorRollupProcessor(
                knowledge_graph=object(),
                domain_by_code=DIAMOND_KG_DOMAINS,
                hierarchy_relation="is_a",
                root="R",
                target_policy="first_retained",
                include_root=False,
            )

    def test_unknown_relation_rejected(self):
        with self.assertRaises(ValueError):
            build_processor(hierarchy_relation="part_of")

    def test_missing_domain_coverage_rejected(self):
        domains = {k: v for k, v in DIAMOND_KG_DOMAINS.items() if k != "N"}
        with self.assertRaises(ValueError):
            build_processor(domains=domains)

    def test_bad_domain_metadata_rejected(self):
        domains = dict(DIAMOND_KG_DOMAINS, N="")
        with self.assertRaises(ValueError):
            build_processor(domains=domains)

    def test_non_mapping_domain_by_code_rejected(self):
        with self.assertRaises(TypeError):
            build_processor(domain_by_code=["Condition"] * 11)

    def test_root_must_be_an_eligible_entity(self):
        with self.assertRaises(ValueError):
            build_processor(root="N")  # Observation
        with self.assertRaises(ValueError):
            build_processor(root="NOPE")

    def test_configuration_validated(self):
        with self.assertRaises(ValueError):
            build_processor(target_policy="nearest")
        with self.assertRaises(ValueError):
            build_processor(max_depth=-1)
        with self.assertRaises(TypeError):
            build_processor(max_depth=True)
        with self.assertRaises(ValueError):
            build_processor(on_unmapped="keep")
        with self.assertRaises(ValueError):
            build_processor(path_source="rescue")

    def test_supplied_depths_validated(self):
        with self.assertRaises(ValueError):
            build_processor(root_depths={"R": 1, "A": 1})
        with self.assertRaises(ValueError):
            build_processor(root_depths={"R": 0, "A": 1, "ZZZ": 1})
        with self.assertRaises(ValueError):
            build_processor(root_depths={"R": 0, "A": -1})
        with self.assertRaises(TypeError):
            build_processor(root_depths={"R": 0, "A": True})

    def test_caller_inputs_not_mutated(self):
        domains = dict(DIAMOND_KG_DOMAINS)
        depths = dict(DISAGREEING_DEPTHS)
        build_processor(domain_by_code=domains, root_depths=depths)
        self.assertEqual(domains, DIAMOND_KG_DOMAINS)
        self.assertEqual(depths, DISAGREEING_DEPTHS)


class TestPathSourceValidation(unittest.TestCase):
    """The closure path source on the generic constructor."""

    # M exists in the KG only through a canary "treats" row, so it has no
    # hierarchy edge; the supplied (closure-like) depths still place it deep.
    CLOSURE_TRIPLES: ClassVar = [("R", "is_a", "A"), ("A", "treats", "M")]
    CLOSURE_DOMAINS: ClassVar = {
        "R": "Condition",
        "A": "Condition",
        "M": "Condition",
    }
    #: Authoritative (closure-like) depths: M sits deep with no direct edge.
    CLOSURE_DEPTHS: ClassVar = {"R": 0, "A": 1, "M": 3}

    def build(self, **overrides):
        values = {
            "knowledge_graph": KnowledgeGraph(triples=self.CLOSURE_TRIPLES),
            "domain_by_code": self.CLOSURE_DOMAINS,
            "hierarchy_relation": "is_a",
            "root": "R",
            "target_policy": "first_retained",
            "include_root": False,
            "max_depth": 2,
            "on_unmapped": "empty",
        }
        values.update(overrides)
        return AncestorRollupProcessor(**values)

    def test_closure_requires_closure_pairs(self):
        with self.assertRaises(ValueError):
            self.build(path_source="closure")

    def test_closure_pairs_require_closure_mode(self):
        with self.assertRaises(ValueError):
            self.build(closure_pairs=[("A", "M")])

    def test_direct_mode_reports_no_target_for_edgeless_deep_source(self):
        processor = self.build(root_depths=self.CLOSURE_DEPTHS)
        result = processor.rollup_codes(["M"])[0]
        self.assertEqual(result.reason, "no_target")
        self.assertEqual(result.targets, ())

    def test_closure_mode_rolls_up_and_flags_the_added_edge(self):
        direct = self.build(root_depths=self.CLOSURE_DEPTHS)
        closure = self.build(
            path_source="closure",
            closure_pairs=[("A", "M")],
            root_depths=self.CLOSURE_DEPTHS,
        )
        result = closure.rollup_codes(["M"])[0]
        self.assertEqual(list(result.targets), ["A"])
        self.assertEqual(result.reason, "rolled_up")
        # same authoritative depths in both modes, different fingerprints
        self.assertEqual(
            direct.rollup_codes(["M"])[0].source_depth,
            result.source_depth,
        )
        self.assertEqual(
            closure.provenance["depth_method"], "supplied_root_depths"
        )
        self.assertNotEqual(direct.fingerprint, closure.fingerprint)
        self.assertEqual(closure.provenance["closure_only_edge_count"], "1")

    def test_closure_pair_overlapping_a_direct_edge_stays_direct(self):
        closure = self.build(
            path_source="closure",
            # (R, A) is a direct edge; only (A, M) is closure-only
            closure_pairs=[("R", "A"), ("A", "M")],
            root_depths=self.CLOSURE_DEPTHS,
        )
        self.assertEqual(closure.provenance["closure_only_edge_count"], "1")

    def test_bad_closure_pairs_rejected(self):
        with self.assertRaises(ValueError):
            self.build(
                path_source="closure",
                closure_pairs=[("R", "R")],  # self-edge
                root_depths=self.CLOSURE_DEPTHS,
            )
        with self.assertRaises(ValueError):
            self.build(
                path_source="closure",
                closure_pairs=[("ZZZ", "M")],  # unknown endpoint
                root_depths=self.CLOSURE_DEPTHS,
            )


class TestGraphValidation(unittest.TestCase):
    """Invalid DAGs are rejected over the selected edge set."""

    def test_cycle_rejected_anywhere(self):
        triples = [
            ("R", "is_a", "A"), ("A", "is_a", "B"), ("B", "is_a", "A"),
        ]
        with self.assertRaises(ValueError):
            build_processor(triples=triples)

    def test_self_edge_rejected(self):
        triples = [("R", "is_a", "A"), ("A", "is_a", "A")]
        with self.assertRaises(ValueError):
            build_processor(triples=triples)

    def test_closure_mode_cycle_rejected(self):
        # Direct edges form a DAG, but the selected closure pair closes one.
        triples = [("R", "is_a", "A"), ("A", "is_a", "B")]
        domains = {"R": "Condition", "A": "Condition", "B": "Condition"}
        with self.assertRaises(ValueError):
            AncestorRollupProcessor(
                knowledge_graph=KnowledgeGraph(triples=triples),
                domain_by_code=domains,
                hierarchy_relation="is_a",
                root="R",
                target_policy="first_retained",
                include_root=False,
                path_source="closure",
                closure_pairs=[("R", "B"), ("B", "R")],
            )


class TestMapping(unittest.TestCase):
    """Mapping behaviour, occurrences, and failure modes (no PyG needed)."""

    SOURCES: ClassVar = ("X", "Y", "C", "U", "R", "N", "O", "W", "MISSING")

    def test_diamond_behaviour_matches_the_reference(self):
        _, processor = build_processor()
        results = processor.rollup_codes(self.SOURCES)
        self.assertEqual(
            [list(r.targets) for r in results],
            [["C", "D"], ["C", "D"], ["C"], ["U"], [], [], [], [], []],
        )
        self.assertEqual(
            [r.reason for r in results],
            ["rolled_up", "rolled_up", "retained", "retained",
             "no_target", "outside_domain", "outside_domain",
             "unreachable", "unknown_code"],
        )
        self.assertEqual(
            [r.source_depth for r in results],
            [3, 4, 2, 1, 0, 1, None, None, None],
        )

    def test_only_the_named_relation_becomes_hierarchy(self):
        _, processor = build_processor()
        # If the "treats" rows leaked in, X would gain parent B (retained at
        # depth 1) and appear as a first_retained target, and W would be
        # reachable instead of unreachable.
        self.assertEqual(
            list(processor.rollup_codes(["X"])[0].targets), ["C", "D"]
        )
        self.assertEqual(processor.rollup_codes(["W"])[0].reason, "unreachable")

    def test_one_result_per_occurrence_in_order(self):
        _, processor = build_processor()
        results = processor.rollup_codes(["X", "C", "X", "MISSING", "X"])
        self.assertEqual(
            [r.source for r in results], ["X", "C", "X", "MISSING", "X"]
        )
        self.assertEqual(
            [list(r.targets) for r in results],
            [["C", "D"], ["C"], ["C", "D"], [], ["C", "D"]],
        )

    def test_repeated_paths_do_not_multiply_targets(self):
        # X reaches C and D through diamond paths and a duplicated edge row;
        # each occurrence still reports each target exactly once.
        _, processor = build_processor()
        self.assertEqual(
            list(processor.rollup_codes(["X"])[0].targets), ["C", "D"]
        )

    def test_empty_input(self):
        _, processor = build_processor()
        self.assertEqual(processor.rollup_codes([]), [])

    def test_error_mode_raises_with_source_and_reason(self):
        _, processor = build_processor(on_unmapped="error")
        expectations = {
            "MISSING": "unknown_code",
            "N": "outside_domain",
            "O": "outside_domain",
            "W": "unreachable",
            "R": "no_target",
        }
        for source, reason in expectations.items():
            with self.assertRaises(UnmappedCodeError) as caught:
                processor.rollup_codes([source])
            self.assertEqual(caught.exception.source, source)
            self.assertEqual(caught.exception.reason, reason)

    def test_empty_mode_returns_typed_reasons(self):
        _, processor = build_processor()
        result = processor.rollup_codes(["MISSING"])[0]
        self.assertIsInstance(result, RollupResult)
        self.assertEqual(result.targets, ())
        self.assertEqual(result.reason, "unknown_code")

    def test_malformed_input_rejected_independently_of_mode(self):
        for mode in ("error", "empty"):
            _, processor = build_processor(on_unmapped=mode)
            with self.assertRaises(TypeError):
                processor.rollup_codes([5])
            with self.assertRaises(ValueError):
                processor.rollup_codes([""])
            with self.assertRaises(TypeError):
                processor.rollup_codes("XC")

    def test_default_mode_is_error(self):
        processor = AncestorRollupProcessor(
            knowledge_graph=KnowledgeGraph(triples=DIAMOND_KG_TRIPLES),
            domain_by_code=DIAMOND_KG_DOMAINS,
            hierarchy_relation="is_a",
            root="R",
            target_policy="first_retained",
            include_root=False,
            max_depth=2,
        )
        with self.assertRaises(UnmappedCodeError):
            processor.rollup_codes(["MISSING"])

    def test_supplied_depth_disagreement_changes_the_mapping(self):
        _, processor = build_processor(root_depths=DISAGREEING_DEPTHS)
        self.assertEqual(list(processor.rollup_codes(["U"])[0].targets), ["D"])
        self.assertEqual(processor.rollup_codes(["U"])[0].reason, "rolled_up")
        self.assertEqual(
            list(processor.rollup_codes(["X"])[0].targets), ["C", "D"]
        )

    def test_supplied_depths_missing_codes_are_unreachable(self):
        # W has no is_a edges and no supplied depth; even though it is a
        # Condition entity, it stays unreachable.
        _, processor = build_processor(root_depths=DISAGREEING_DEPTHS)
        self.assertEqual(processor.rollup_codes(["W"])[0].reason, "unreachable")

    def test_retained_identity_uses_the_supplied_depth(self):
        _, processor = build_processor(root_depths=DISAGREEING_DEPTHS)
        result = processor.rollup_codes(["A"])[0]
        self.assertEqual(result.reason, "retained")
        self.assertEqual(result.targets, ("A",))
        self.assertEqual(result.source_depth, 1)


@unittest.skipUnless(_has_pyg(), "torch-geometric not installed")
class TestProjection(unittest.TestCase):
    """Exact projected edges, masks, containers, and empty graphs."""

    def test_graph_embeddings_project_and_batch_with_empty_samples(self):
        from torch_geometric.data import Batch

        vectors = torch.arange(len(DIAMOND_KG_DOMAINS) * 3).reshape(-1, 3).float()
        graph = KnowledgeGraph(DIAMOND_KG_TRIPLES, node_features=vectors)
        _, processor = build_processor(knowledge_graph=graph)
        changed_graph = KnowledgeGraph(DIAMOND_KG_TRIPLES, node_features=vectors + 1)
        _, changed = build_processor(knowledge_graph=changed_graph)
        self.assertNotEqual(processor.fingerprint, changed.fingerprint)

        populated = processor.process(["X"])
        empty = processor.process(["MISSING"])
        self.assertTrue(torch.equal(populated.x, vectors[populated.node_ids]))
        self.assertEqual(tuple(empty.x.shape), (0, 3))
        batch = Batch.from_data_list([empty, populated])
        self.assertTrue(torch.equal(batch.x, populated.x))
        restored = pickle.loads(pickle.dumps(processor))
        self.assertTrue(torch.equal(restored.process(["X"]).x, populated.x))

    def test_identity_projection_on_ontology_without_edges(self):
        graph = KnowledgeGraph(
            triples=[], entity2id={"R": 0}, relation2id={"is_a": 0}
        )
        _, processor = build_processor(
            knowledge_graph=graph, domains={"R": "Condition"}, include_root=True
        )
        data = processor.process(["R"])
        self.assertEqual(data.node_ids.tolist(), [0])
        self.assertEqual(tuple(data.edge_index.shape), (2, 0))

    def test_diamond_x_keeps_both_parent_edges(self):
        values, processor = build_processor()
        data = processor.process(["X"])
        self.assertEqual(node_codes(processor, data), ("C", "D", "X"))
        self.assertEqual(edge_pairs(processor, data), (("C", "X"), ("D", "X")))
        self.assertEqual(
            data.edge_index.tolist(), [[0, 1], [2, 2]]
        )
        relation_id = values["knowledge_graph"].relation2id["is_a"]
        self.assertEqual(data.edge_type.tolist(), [relation_id] * 2)
        self.assertEqual(
            data.node_ids.tolist(),
            [values["knowledge_graph"].entity2id[c] for c in ("C", "D", "X")],
        )
        self.assertEqual(data.seed_mask.tolist(), [False, False, True])
        self.assertEqual(data.target_mask.tolist(), [True, True, False])
        self.assertEqual(data.closure_only.tolist(), [False, False])
        self.assertEqual(tuple(data.x.shape), (3, 1))
        self.assertTrue(torch.equal(data.x, torch.zeros(3, 1)))
        self.assertEqual((data.num_nodes, data.num_edges), (3, 2))

    def test_deep_descendant_includes_intermediate_edges(self):
        _, processor = build_processor()
        data = processor.process(["Y"])
        self.assertEqual(node_codes(processor, data), ("C", "D", "X", "Y"))
        self.assertEqual(
            edge_pairs(processor, data),
            (("C", "X"), ("D", "X"), ("X", "Y")),
        )
        # X is a traversed intermediate: neither a seed nor a target.
        self.assertEqual(data.seed_mask.tolist(), [False, False, False, True])
        self.assertEqual(data.target_mask.tolist(), [True, True, False, False])

    def test_union_across_sources(self):
        _, processor = build_processor()
        data = processor.process(["X", "Y"])
        self.assertEqual(
            edge_pairs(processor, data),
            (("C", "X"), ("D", "X"), ("X", "Y")),
        )
        self.assertEqual(data.seed_mask.tolist(), [False, False, True, True])

    def test_duplicate_occurrences_merge_in_the_graph_only(self):
        _, processor = build_processor()
        single = processor.process(["X"])
        doubled = processor.process(["X", "X"])
        self.assertEqual(
            edge_pairs(processor, doubled), edge_pairs(processor, single)
        )
        self.assertEqual(
            node_codes(processor, doubled), node_codes(processor, single)
        )
        # the ordered mapping still reports one result per occurrence
        self.assertEqual(
            [r.source for r in processor.rollup_codes(["X", "X"])], ["X", "X"]
        )

    def test_identity_mapping_is_a_node_without_edges(self):
        _, processor = build_processor()
        for code in ("C", "U"):
            data = processor.process([code])
            self.assertEqual(node_codes(processor, data), (code,))
            self.assertEqual(edge_pairs(processor, data), ())
            self.assertEqual(data.seed_mask.tolist(), [True])
            self.assertEqual(data.target_mask.tolist(), [True])
            self.assertEqual(data.closure_only.tolist(), [])

    def test_all_retained_projects_all_successful_paths(self):
        _, processor = build_processor(target_policy="all_retained")
        data = processor.process(["X"])
        self.assertEqual(
            node_codes(processor, data), ("A", "B", "C", "D", "X")
        )
        self.assertEqual(
            edge_pairs(processor, data),
            (("A", "C"), ("B", "D"), ("C", "X"), ("D", "X")),
        )
        self.assertEqual(
            data.target_mask.tolist(), [True, True, True, True, False]
        )

    def test_root_inclusion_adds_root_edges(self):
        _, processor = build_processor(
            target_policy="all_retained", include_root=True
        )
        data = processor.process(["X"])
        self.assertEqual(
            node_codes(processor, data), ("A", "B", "C", "D", "R", "X")
        )
        self.assertEqual(
            edge_pairs(processor, data),
            (("A", "C"), ("B", "D"), ("C", "X"), ("D", "X"),
             ("R", "A"), ("R", "B")),
        )
        self.assertTrue(data.target_mask.tolist()[4])  # the root is a target

    def test_supplied_depth_rolls_u_up_with_dead_end_excluded(self):
        # Supplied depths make U (BFS depth 1) a deep code: it rolls up to D.
        # The R branch dead-ends (the excluded root has no parents), so only
        # the successful (D, U) edge is projected.
        _, processor = build_processor(root_depths=DISAGREEING_DEPTHS)
        data = processor.process(["U"])
        self.assertEqual(node_codes(processor, data), ("D", "U"))
        self.assertEqual(edge_pairs(processor, data), (("D", "U"),))
        self.assertEqual(data.seed_mask.tolist(), [False, True])
        self.assertEqual(data.target_mask.tolist(), [True, False])

    def test_non_condition_intermediate_is_traversed_but_never_projected(self):
        # S has a productive branch through retained A and a dead-end branch
        # through the Observation intermediate Z up to the excluded root.
        triples = [
            ("R", "is_a", "A"), ("R", "is_a", "Z"), ("A", "is_a", "S"),
            ("Z", "is_a", "S"),
        ]
        domains = {"R": "Condition", "A": "Condition", "Z": "Observation",
                   "S": "Condition"}
        _, processor = build_processor(
            triples=triples, domains=domains, max_depth=1
        )
        self.assertEqual(
            list(processor.rollup_codes(["S"])[0].targets), ["A"]
        )
        data = processor.process(["S"])
        self.assertEqual(node_codes(processor, data), ("A", "S"))
        self.assertEqual(edge_pairs(processor, data), (("A", "S"),))
        self.assertNotIn(("Z", "S"), edge_pairs(processor, data))

    def test_closure_mode_flags_only_the_closure_edge(self):
        # The generic reproduction of the recorded Athena discrepancy: M is
        # deep with no hierarchy edge; the closure pair stitches it to A.
        triples = [("R", "is_a", "A"), ("A", "treats", "M")]
        domains = {"R": "Condition", "A": "Condition", "M": "Condition"}
        depths = {"R": 0, "A": 1, "M": 3}
        processor = AncestorRollupProcessor(
            knowledge_graph=KnowledgeGraph(triples=triples),
            domain_by_code=domains,
            hierarchy_relation="is_a",
            root="R",
            target_policy="first_retained",
            include_root=False,
            max_depth=2,
            on_unmapped="empty",
            root_depths=depths,
            path_source="closure",
            closure_pairs=[("A", "M")],
        )
        data = processor.process(["M"])
        self.assertEqual(node_codes(processor, data), ("A", "M"))
        self.assertEqual(edge_pairs(processor, data), (("A", "M"),))
        self.assertEqual(data.closure_only.tolist(), [True])
        self.assertEqual(data.seed_mask.tolist(), [False, True])
        self.assertEqual(data.target_mask.tolist(), [True, False])

    def test_unmapped_only_sample_is_a_consistently_shaped_empty_graph(self):
        _, processor = build_processor()
        data = processor.process(["MISSING"])
        self.assertEqual(data.num_nodes, 0)
        self.assertEqual(data.num_edges, 0)
        self.assertEqual(data.edge_index.shape, (2, 0))
        self.assertEqual(data.edge_type.tolist(), [])
        self.assertEqual(data.node_ids.tolist(), [])
        self.assertEqual(data.x.shape, (0, 1))
        self.assertEqual(data.seed_mask.tolist(), [])
        self.assertEqual(data.target_mask.tolist(), [])
        self.assertEqual(data.closure_only.tolist(), [])

    def test_empty_inputs_yield_empty_graphs(self):
        _, processor = build_processor()
        for value in ([], [[]]):
            data = processor.process(value)
            self.assertEqual(data.num_nodes, 0)
            self.assertEqual(data.num_edges, 0)

    def test_unmapped_codes_are_excluded_from_a_mixed_sample(self):
        _, processor = build_processor()
        mixed = processor.process(["MISSING", "X"])
        only_x = processor.process(["X"])
        self.assertEqual(
            edge_pairs(processor, mixed), edge_pairs(processor, only_x)
        )
        self.assertEqual(
            node_codes(processor, mixed), node_codes(processor, only_x)
        )

    def test_error_mode_failure_inside_a_sample_raises(self):
        _, processor = build_processor(on_unmapped="error")
        with self.assertRaises(UnmappedCodeError) as caught:
            processor.process(["X", "MISSING"])
        self.assertEqual(caught.exception.source, "MISSING")

    def test_visits_flatten_in_order(self):
        _, processor = build_processor()
        data = processor.process([["X"], ["Y", "C"]])
        self.assertEqual(
            [r.source for r in processor.rollup_codes(["X", "Y", "C"])],
            ["X", "Y", "C"],
        )
        self.assertEqual(node_codes(processor, data), ("C", "D", "X", "Y"))
        # C was observed this visit, so it is both a seed and a target.
        self.assertEqual(data.seed_mask.tolist(), [True, False, True, True])
        self.assertEqual(data.target_mask.tolist(), [True, True, False, False])

    def test_flat_and_visit_graphs_agree(self):
        _, processor = build_processor()
        flat = processor.process(["X", "Y", "C"])
        visits = processor.process([["X"], ["Y", "C"]])
        self.assertEqual(
            edge_pairs(processor, flat), edge_pairs(processor, visits)
        )
        self.assertEqual(
            node_codes(processor, flat), node_codes(processor, visits)
        )
        self.assertEqual(
            flat.seed_mask.tolist(), visits.seed_mask.tolist()
        )

    def test_malformed_sample_input_rejected(self):
        _, processor = build_processor()
        with self.assertRaises(TypeError):
            processor.process("X")
        with self.assertRaises(TypeError):
            processor.process(5)
        with self.assertRaises(TypeError):
            processor.process([["X"], "Y"])
        with self.assertRaises(ValueError):
            processor.process(["X", ""])


@unittest.skipUnless(_has_pyg(), "torch-geometric not installed")
class TestBatchingAndDisk(unittest.TestCase):
    """Standard PyG batching and disk round-tripping."""

    def test_batching_concatenates_samples_and_reindexes_edges(self):
        from torch_geometric.data import Batch

        _, processor = build_processor()
        first = processor.process(["X"])
        second = processor.process(["Y"])
        batch = Batch.from_data_list([first, second])
        self.assertEqual(batch.num_nodes, first.num_nodes + second.num_nodes)
        self.assertEqual(
            batch.num_edges, first.num_edges + second.num_edges
        )
        # local edge indices are offset per sample and stay in range
        self.assertLess(
            int(batch.edge_index.max()), batch.num_nodes
        )
        # global entity ids concatenate without offsetting
        self.assertEqual(
            batch.node_ids.tolist(),
            first.node_ids.tolist() + second.node_ids.tolist(),
        )
        self.assertEqual(
            batch.seed_mask.tolist(),
            first.seed_mask.tolist() + second.seed_mask.tolist(),
        )
        self.assertEqual(
            batch.closure_only.tolist(),
            first.closure_only.tolist() + second.closure_only.tolist(),
        )

    def test_disk_round_trip_preserves_all_fields(self):
        _, processor = build_processor()
        data = processor.process(["X", "Y"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "graph.pt"
            torch.save(data, path)
            loaded = torch.load(path, weights_only=False)
        self.assertEqual(
            edge_pairs(processor, loaded), edge_pairs(processor, data)
        )
        self.assertEqual(
            node_codes(processor, loaded), node_codes(processor, data)
        )
        self.assertEqual(
            loaded.seed_mask.tolist(), data.seed_mask.tolist()
        )
        self.assertEqual(
            loaded.target_mask.tolist(), data.target_mask.tolist()
        )
        self.assertEqual(
            loaded.closure_only.tolist(), data.closure_only.tolist()
        )


@unittest.skipUnless(_has_pyg(), "torch-geometric not installed")
class TestDocstringExamples(unittest.TestCase):
    """The documented examples run as written."""

    def test_module_and_class_docstrings(self):
        import doctest

        from pyhealth.processors import ancestor_rollup_processor

        results = doctest.testmod(
            ancestor_rollup_processor, verbose=False, optionflags=0
        )
        self.assertEqual(results.failed, 0)
        self.assertGreater(results.attempted, 0)


if __name__ == "__main__":
    unittest.main()
