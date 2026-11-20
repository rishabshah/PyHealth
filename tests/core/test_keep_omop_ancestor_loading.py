"""Checks which ancestor pairs the KEEP factory materializes."""

import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from pyhealth.medcode import OMOPVocabulary
from pyhealth.tasks import build_keep_omop_ancestor_rollup
from tests.core.test_ancestor_rollup_from_omop import ROOT, write_athena_bundle


class TestKeepOmopAncestorLoading(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        write_athena_bundle(folder)
        self.vocabulary = OMOPVocabulary(folder)

    def collect_ancestor_pairs(self, path_source):
        batches = []
        original_collect = pl.LazyFrame.collect

        def record_collect(frame, *args, **kwargs):
            result = original_collect(frame, *args, **kwargs)
            # The root-depth query has no ancestor_id column. Only the
            # traversal-pair query emits a batch of ancestor endpoints.
            if {"ancestor_id", "concept_id"} <= set(result.columns):
                batches.append(
                    list(zip(result["ancestor_id"], result["concept_id"], strict=True))
                )
            return result

        with patch.object(
            pl.LazyFrame, "collect", autospec=True, side_effect=record_collect
        ):
            processor = build_keep_omop_ancestor_rollup(
                self.vocabulary,
                target_policy="first_retained",
                include_root=False,
                max_depth=3,
                on_unmapped="empty",
                path_source=path_source,
            )

        return processor, batches

    def test_direct_loads_depths_without_materializing_traversal_pairs(self):
        processor, batches = self.collect_ancestor_pairs("direct")

        self.assertEqual(batches, [])
        result = processor.rollup_codes(["103"])[0]
        self.assertEqual(result.source_depth, 4)
        self.assertEqual(result.targets, ("102",))

    def test_closure_materializes_only_separation_one_pairs_within_subtree(self):
        processor, batches = self.collect_ancestor_pairs("closure")

        self.assertEqual(len(batches), 1)
        self.assertCountEqual(
            batches[0],
            [
                (ROOT, 100),
                (ROOT, 111),
                (100, 101),
                (100, 104),
                (101, 102),
                (102, 103),
                (103, 104),
                (103, 107),
                (111, 222),
                (222, 333),
                (333, 140708),
                (140708, 107),
                (140708, 1075201),
            ],
        )
        self.assertEqual(processor.rollup_codes(["1075201"])[0].targets, ("333",))


if __name__ == "__main__":
    unittest.main()
