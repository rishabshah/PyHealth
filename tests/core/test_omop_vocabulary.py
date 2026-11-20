"""Tests for pyhealth.medcode.omop.vocabulary (the Athena reader).

Uses the synthetic bundle in ``omop_test_utils``; no real Athena files.
"""

import tempfile
import unittest
from pathlib import Path

import polars as pl

from pyhealth.medcode import OMOPVocabulary, OMOPVocabularyFormat

from tests.core.omop_test_utils import AthenaFixture, write_athena_bundle


class TestOMOPVocabulary(AthenaFixture):
    def test_missing_folder(self):
        with self.assertRaises(FileNotFoundError):
            OMOPVocabulary(Path(self.tmp) / "nope")

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            write_athena_bundle(folder)
            (Path(folder) / "CONCEPT_ANCESTOR.csv").unlink()
            with self.assertRaises(FileNotFoundError):
                OMOPVocabulary(folder)

    def test_comma_separated_file_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            write_athena_bundle(folder, separator=",")
            with self.assertRaisesRegex(ValueError, "tab-separated"):
                OMOPVocabulary(folder)

    def test_scan_reads_ids_as_integers_and_keeps_quotes(self):
        concept = self.vocab.scan("CONCEPT").collect()
        self.assertEqual(concept.schema["concept_id"], pl.Int64)
        names = dict(concept.select("concept_id", "concept_name").iter_rows())
        self.assertEqual(names[102], 'Acute "MI')

    def test_unknown_table(self):
        with self.assertRaises(ValueError):
            self.vocab.scan("DRUG_STRENGTH")


class TestOMOPVocabularyFormat(AthenaFixture):
    def test_defaults_match_athena(self):
        self.assertEqual(self.vocab.file_format, OMOPVocabularyFormat())
        self.assertEqual(OMOPVocabularyFormat().separator, "\t")
        self.assertIsNone(OMOPVocabularyFormat().quote_char)

    def test_comma_separated_export_with_other_file_names(self):
        file_format = OMOPVocabularyFormat(
            separator=",", quote_char='"', file_name="vocab_{table}.txt"
        )
        with tempfile.TemporaryDirectory() as folder:
            write_athena_bundle(folder, separator=",", file_name="vocab_{table}.txt")
            vocab = OMOPVocabulary(folder, file_format=file_format)
            hierarchy = vocab.condition_hierarchy(max_depth=3)
            mapper = vocab.icd_mapper()
            names = dict(
                vocab.scan("CONCEPT")
                .collect()
                .select("concept_id", "concept_name")
                .iter_rows()
            )
        # same results as the Athena-format bundle
        self.assertTrue(hierarchy.nodes.equals(self.hierarchy.nodes))
        self.assertTrue(hierarchy.edges.equals(self.hierarchy.edges))
        self.assertEqual(mapper.map("I2510", 10), [101, 104])
        # commas and quotes inside quoted fields survive
        self.assertEqual(names[103], "Acute MI, inferior wall")
        self.assertEqual(names[102], 'Acute "MI')

    def test_invalid_values(self):
        for kwargs in (
            {"separator": "::"},
            {"quote_char": "''"},
            {"encoding": "latin-1"},
            {"file_name": "CONCEPT.csv"},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                OMOPVocabularyFormat(**kwargs)


if __name__ == "__main__":
    unittest.main()
