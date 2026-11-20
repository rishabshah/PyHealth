"""Tests for pyhealth.medcode.omop.icd_mapper.

Uses the synthetic bundle in ``omop_test_utils``; no real Athena or MIMIC data.
"""

import tempfile
import unittest
from pathlib import Path

import polars as pl

from pyhealth.medcode import ICDToSNOMEDMapper
from pyhealth.medcode.omop import normalize_icd_code

from tests.core.omop_test_utils import AthenaFixture


class TestICDToSNOMEDMapper(AthenaFixture):
    def test_normalize(self):
        self.assertEqual(normalize_icd_code(" i21.4 "), "I214")
        self.assertEqual(normalize_icd_code("E880.9"), "E8809")

    def test_map_mimic_and_dotted_codes(self):
        self.assertEqual(self.mapper.map("I214", 10), [102])
        self.assertEqual(self.mapper.map("I21.4", "10"), [102])
        self.assertEqual(self.mapper.map("41071", 9), [102])
        self.assertEqual(self.mapper.map("E8809", 9), [104])
        self.assertEqual(self.mapper.map("V4581", 9), [101])

    def test_one_code_several_concepts(self):
        self.assertEqual(self.mapper.map("I2510", 10), [101, 104])

    def test_unmapped_unknown_and_deprecated(self):
        self.assertEqual(self.mapper.map("Z9989", 10), [])  # no "Maps to"
        self.assertIn((10, "Z9989"), self.mapper)
        self.assertEqual(self.mapper.map("XYZ", 10), [])  # not in Athena
        self.assertNotIn((10, "XYZ"), self.mapper)
        # deprecated source code still maps through a valid link
        self.assertEqual(self.mapper.map("I219", 10), [102])

    def test_versions_are_separate(self):
        self.assertEqual(self.mapper.map("I214", 9), [])
        with self.assertRaises(ValueError):
            self.mapper.map("I214", 11)
        icd10_only = self.vocab.icd_mapper(versions=[10])
        self.assertEqual(icd10_only.map("41071", 9), [])

    def test_map_frame(self):
        diagnoses = pl.DataFrame(
            {
                "subject_id": [1, 1, 2],
                "icd_code": ["I214", "I2510", "Z9989"],
                "icd_version": [10, 10, 10],
            }
        )
        mapped = self.mapper.map_frame(diagnoses)
        self.assertEqual(mapped.height, 4)  # I2510 maps to two concepts
        self.assertEqual(
            sorted(mapped.filter(pl.col("icd_code") == "I2510")["concept_id"]),
            [101, 104],
        )
        self.assertIsNone(
            mapped.filter(pl.col("icd_code") == "Z9989")["concept_id"][0]
        )
        self.assertEqual(mapped.columns[:3], diagnoses.columns)

    def test_coverage(self):
        diagnoses = pl.DataFrame(
            {
                "icd_code": ["I214", "I214", "I2510", "R509", "Z9989", "XYZ",
                             "41071", "E8809"],
                "icd_version": [10, 10, 10, 10, 10, 10, 9, 9],
            }
        )
        self.assertEqual(
            self.mapper.coverage(diagnoses, hierarchy=self.hierarchy),
            {
                "codes": 7,
                "unknown": 1,
                "no_mapping": 1,
                "mapped": 5,
                "in_hierarchy": 4,
                "outside_hierarchy": 1,  # R50.9 -> Fever, not a disease
            },
        )

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "icd_to_snomed.parquet"
            self.mapper.save(path)
            loaded = ICDToSNOMEDMapper.load(path)
        self.assertEqual(loaded.map("I2510", 10), [101, 104])

    def test_map_frame_keeps_lazy_frames_lazy(self):
        diagnoses = pl.LazyFrame(
            {"icd_code": ["I214", "I2510"], "icd_version": [10, 10]}
        )
        mapped = self.mapper.map_frame(diagnoses)
        self.assertIsInstance(mapped, pl.LazyFrame)
        self.assertEqual(
            sorted(mapped.collect()["concept_id"].to_list()), [101, 102, 104]
        )

    def test_coverage_of_a_lazy_frame(self):
        eager = pl.DataFrame(
            {"icd_code": ["I214", "Z9989", "XYZ"], "icd_version": [10, 10, 10]}
        )
        self.assertEqual(
            self.mapper.coverage(eager.lazy(), hierarchy=self.hierarchy),
            self.mapper.coverage(eager, hierarchy=self.hierarchy),
        )


if __name__ == "__main__":
    unittest.main()
