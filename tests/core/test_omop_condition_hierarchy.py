"""Tests for pyhealth.medcode.omop.condition_hierarchy.

Uses the synthetic bundle in ``omop_test_utils``; see its docstring for the
concept tree. The GraphProcessor tests need torch-geometric and are skipped
without it.
"""

import tempfile
import unittest

import polars as pl

from pyhealth.medcode import OMOPConditionHierarchy

from tests.core.omop_test_utils import ROOT, AthenaFixture

try:
    from pyhealth.processors import GraphProcessor
    import torch_geometric  # noqa: F401

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class TestOMOPConditionHierarchy(AthenaFixture):
    def test_nodes_respect_depth_domain_and_validity(self):
        self.assertEqual(
            set(self.hierarchy.nodes["concept_id"]),
            {ROOT, 100, 200, 101, 102, 104, 105},
        )
        for excluded in (103, 600, 300, 400, 401, 700):
            self.assertNotIn(excluded, self.hierarchy)

    def test_depths(self):
        self.assertEqual(self.hierarchy.depth(ROOT), 0)
        self.assertEqual(self.hierarchy.depth(102), 3)
        # two parents: shortest path (via 100) is 2, longest (via 103) is 5
        row = self.hierarchy.nodes.filter(pl.col("concept_id") == 104).row(
            0, named=True
        )
        self.assertEqual((row["depth"], row["min_depth"], row["max_depth"]),
                         (2, 2, 5))

    def test_max_depth_measure(self):
        deepest = self.vocab.condition_hierarchy(max_depth=3, depth_measure="max")
        self.assertNotIn(104, deepest)
        self.assertIn(102, deepest)

    def test_no_depth_limit(self):
        full = self.vocab.condition_hierarchy(max_depth=None)
        self.assertIn(103, full)
        self.assertEqual(full.parents(104), [100, 103])

    def test_edges_only_between_kept_concepts(self):
        edges = set(
            self.hierarchy.edges.filter(~pl.col("rescued"))
            .select("parent_id", "child_id")
            .iter_rows()
        )
        self.assertEqual(
            edges, {(ROOT, 100), (ROOT, 200), (100, 101), (101, 102), (100, 104)}
        )
        # the deprecated 102 -> 200 link is not an edge
        self.assertEqual(self.hierarchy.parents(102), [101])

    def test_orphan_is_attached_to_nearest_kept_ancestor(self):
        self.assertEqual(self.hierarchy.parents(105), [100])
        rescued = self.hierarchy.edges.filter(pl.col("rescued"))
        self.assertEqual(list(rescued.select("parent_id", "child_id").iter_rows()),
                         [(100, 105)])

    def test_orphans_left_alone_when_asked(self):
        loose = self.vocab.condition_hierarchy(max_depth=3, connect_orphans=False)
        self.assertEqual(loose.parents(105), [])
        self.assertFalse(loose.edges["rescued"].any())

    def test_retained_ancestors_of_deep_and_multi_parent_concepts(self):
        # 103 is below the cutoff; roll-up needs its kept ancestors
        self.assertEqual(
            self.hierarchy.retained_ancestors(103),
            [(102, 1), (101, 2), (100, 3), (ROOT, 4)],
        )
        self.assertEqual(
            self.hierarchy.retained_ancestors(104),
            [(100, 1), (102, 2), (ROOT, 2), (101, 3)],
        )
        self.assertEqual(self.hierarchy.retained_ancestors(300), [])

    def test_lookups(self):
        self.assertEqual(self.hierarchy.name(ROOT), "Disease")
        self.assertEqual(self.hierarchy.children(ROOT), [100, 200])
        with self.assertRaises(KeyError):
            self.hierarchy.depth(103)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            self.vocab.condition_hierarchy(depth_measure="mean")
        with self.assertRaises(ValueError):
            self.vocab.condition_hierarchy(max_depth=-1)
        with self.assertRaises(ValueError):
            self.vocab.condition_hierarchy(root_concept_id=600)

    def test_networkx(self):
        graph = self.hierarchy.to_networkx()
        self.assertEqual(graph.number_of_nodes(), len(self.hierarchy))
        self.assertEqual(graph.nodes[102]["depth"], 3)
        self.assertTrue(graph.edges[100, 105]["rescued"])

    def test_restrict_matches_direct_build(self):
        full = self.vocab.condition_hierarchy(max_depth=None)
        for depth, measure in [(3, "min"), (2, "min"), (3, "max"), (0, "min")]:
            direct = self.vocab.condition_hierarchy(
                max_depth=depth, depth_measure=measure
            )
            cut = full.restrict(depth, measure)
            with self.subTest(depth=depth, measure=measure):
                self.assertTrue(cut.nodes.equals(direct.nodes))
                self.assertTrue(cut.edges.equals(direct.edges))
                self.assertTrue(
                    cut.ancestors.sort(cut.ancestors.columns).equals(
                        direct.ancestors.sort(direct.ancestors.columns)
                    )
                )
                self.assertEqual((cut.max_depth, cut.depth_measure),
                                 (depth, measure))

    def test_restrict_refuses_to_add_concepts(self):
        with self.assertRaises(ValueError):
            self.hierarchy.restrict(4)
        by_max = self.vocab.condition_hierarchy(max_depth=3, depth_measure="max")
        with self.assertRaises(ValueError):
            by_max.restrict(3, "min")
        self.assertEqual(len(self.hierarchy.restrict(3, "max")),
                         len(by_max))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            self.hierarchy.save(folder)
            loaded = OMOPConditionHierarchy.load(folder)
        self.assertEqual(loaded.max_depth, 3)
        self.assertEqual(loaded.depth_measure, "min")
        self.assertTrue(loaded.nodes.equals(self.hierarchy.nodes))
        self.assertEqual(loaded.retained_ancestors(103),
                         self.hierarchy.retained_ancestors(103))


class TestTriples(AthenaFixture):
    def test_child_to_parent(self):
        triples = self.hierarchy.to_triples()
        self.assertEqual(triples, self.hierarchy.to_triples("child_to_parent"))
        self.assertIn(("101", "is_a", "100"), triples)
        self.assertEqual(len(triples), self.hierarchy.edges.height)
        self.assertTrue(all(isinstance(part, str) for t in triples for part in t))

    def test_parent_to_child(self):
        triples = self.hierarchy.to_triples("parent_to_child")
        self.assertIn(("100", "subsumes", "101"), triples)
        self.assertNotIn(("101", "is_a", "100"), triples)

    def test_rescue_edges_keep_their_own_relation(self):
        up = self.hierarchy.to_triples("child_to_parent")
        down = self.hierarchy.to_triples("parent_to_child")
        # 105's only parent was in another domain; it was attached to 100,
        # two levels up, so the edge must not look like a direct link
        self.assertIn(("105", "rescued_is_a", "100"), up)
        self.assertNotIn(("105", "is_a", "100"), up)
        self.assertIn(("100", "rescued_subsumes", "105"), down)
        self.assertEqual(sum(r == "rescued_is_a" for _, r, _ in up), 1)

    def test_bad_direction(self):
        with self.assertRaises(ValueError):
            self.hierarchy.to_triples("sideways")


class TestKnowledgeGraph(AthenaFixture):
    def test_entities_are_string_concept_ids(self):
        kg = self.hierarchy.to_knowledge_graph()
        self.assertEqual(
            set(kg.entity2id), {str(c) for c in self.hierarchy.nodes["concept_id"]}
        )
        self.assertEqual(kg.num_triples, self.hierarchy.edges.height)
        # depth and distances stay on the hierarchy, not the graph
        self.assertFalse(hasattr(kg, "depth"))
        self.assertEqual(self.hierarchy.depth(102), 3)

    def test_direction_of_edges(self):
        down = self.hierarchy.to_knowledge_graph()  # parent_to_child
        up = self.hierarchy.to_knowledge_graph("child_to_parent")
        edge = (down.entity2id["100"], down.entity2id["101"])
        down_edges = set(map(tuple, down.edge_index.t().tolist()))
        up_edges = set(map(tuple, up.edge_index.t().tolist()))
        self.assertIn(edge, down_edges)
        self.assertIn(edge[::-1], up_edges)
        self.assertEqual(set(down.relation2id), {"subsumes", "rescued_subsumes"})
        self.assertEqual(set(up.relation2id), {"is_a", "rescued_is_a"})

    def test_edge_type_marks_rescue_edges(self):
        kg = self.hierarchy.to_knowledge_graph("child_to_parent")
        rescued_type = kg.relation2id["rescued_is_a"]
        rescued = [
            (kg.id2entity[h], kg.id2entity[t])
            for (h, t), r in zip(kg.edge_index.t().tolist(), kg.edge_type.tolist())
            if r == rescued_type
        ]
        self.assertEqual(rescued, [("105", "100")])

    def test_isolated_concepts_are_kept(self):
        loose = self.vocab.condition_hierarchy(max_depth=3, connect_orphans=False)
        kg = loose.to_knowledge_graph()
        self.assertTrue(kg.has_entity("105"))  # no edges, still an entity
        self.assertEqual(kg.num_entities, len(loose))

    def test_no_edges(self):
        with self.assertRaises(ValueError):
            self.vocab.condition_hierarchy(max_depth=0).to_knowledge_graph()


@unittest.skipUnless(HAS_PYG, "torch-geometric required")
class TestGraphProcessor(AthenaFixture):
    """GraphProcessor's k-hop extraction follows edges into the seed codes."""

    def _subgraph_concepts(self, direction, codes, num_hops):
        kg = self.hierarchy.to_knowledge_graph(direction)
        graph = GraphProcessor(knowledge_graph=kg, num_hops=num_hops).process(codes)
        return {kg.id2entity[i] for i in graph.node_ids.tolist()}

    def test_parent_to_child_reaches_ancestors_from_a_child_seed(self):
        self.assertEqual(
            self._subgraph_concepts("parent_to_child", ["102"], num_hops=3),
            {"102", "101", "100", str(ROOT)},
        )

    def test_child_to_parent_does_not_reach_ancestors(self):
        # 102 has no kept children, so only the seed comes back
        self.assertEqual(
            self._subgraph_concepts("child_to_parent", ["102"], num_hops=3),
            {"102"},
        )


if __name__ == "__main__":
    unittest.main()
