"""KEEP history from MIMIC4Dataset, the real mapper, and synthetic rows."""

import importlib.util
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.datasets.configs.config import load_yaml_config
from pyhealth.medcode import DISEASE_CONCEPT_ID, OMOPVocabulary
from pyhealth.tasks import KeepPatientHistory

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "pyhealth" / "datasets" / "configs"


def _cache_dir(tmp_path: Path, name: str = "keep_history") -> Path:
    """Return a cache directory Dask can use.

    PyHealth keeps Dask's scratch space under the cache. Under pytest's
    temp folder on Windows that path exceeds MAX_PATH and the shuffle fails.
    A short directory on the same drive stays under the limit. Other
    platforms keep the pytest temp directory.
    """
    if os.name != "nt":
        return tmp_path / name
    # The repo cache is gitignored and short enough for Dask's scratch files.
    short = REPO / ".pyhealth_cache" / name
    if short.exists():
        shutil.rmtree(short)
    short.mkdir(parents=True)
    return short


def _example():
    path = REPO / "examples" / "mimic_omop_patient_history.py"
    spec = importlib.util.spec_from_file_location("mimic_history_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_omop_config_does_not_require_workflow_columns():
    shared = load_yaml_config(str(CONFIGS / "omop.yaml"))
    interchange = load_yaml_config(str(CONFIGS / "mimic_omop.yaml"))
    shared_attrs = shared.tables["condition_occurrence"].attributes
    interchange_attrs = interchange.tables["condition_occurrence"].attributes
    assert "condition_status_concept_id" not in shared_attrs
    assert "condition_occurrence_id" not in shared_attrs
    assert "condition_status_concept_id" not in interchange_attrs
    assert "condition_source_concept_id" not in interchange_attrs
    assert "condition_occurrence_id" in interchange_attrs
    assert "condition_source_value" in interchange_attrs


def _histories(mimic_root: Path, vocabulary, cache_dir: Path):
    dataset = MIMIC4Dataset(
        ehr_root=str(mimic_root),
        ehr_tables=["diagnoses_icd"],
        ehr_config_path=str(CONFIGS / "mimic4_keep_ehr.yaml"),
        cache_dir=str(cache_dir),
    )
    return dataset.set_task(KeepPatientHistory(vocabulary), num_workers=1)


def test_mimic4_history_covers_the_keep_contract(tmp_path):
    example = _example()
    mimic_root = tmp_path / "mimic"
    athena_root = tmp_path / "athena"
    example.write_synthetic_mimic(mimic_root)
    example.write_synthetic_athena(athena_root)

    vocabulary = OMOPVocabulary(athena_root)
    mapper = vocabulary.icd_mapper()
    assert mapper.map("04119", 9) == [100]
    assert mapper.map("4119", 9) == []
    assert mapper.map("I2510", 10) == [101, 104]
    assert mapper.map("I214", 10) == [102]
    assert mapper.map("41071", 9) == [102]

    cache_dir = _cache_dir(tmp_path)
    try:
        histories = _histories(mimic_root, vocabulary, cache_dir)
        _assert_history(histories)
    finally:
        if cache_dir != tmp_path / "cache":
            shutil.rmtree(cache_dir, ignore_errors=True)


def _assert_history(histories):
    assert len(histories) == 1
    history = histories[0]
    assert history["patient_id"] == "1"
    conditions = {row["diagnosis_id"]: row for row in history["conditions"]}

    # Eight source rows: the duplicate 04119 collapses, leaving six diagnoses.
    assert set(conditions) == {"10:1", "10:2", "11:1", "11:2", "99:1", "10:5"}

    leading_zero = conditions["10:1"]
    assert leading_zero["icd_code"] == "04119"
    assert leading_zero["icd_version"] == 9
    assert leading_zero["concept_ids"] == [100]
    assert leading_zero["conflict"] is False
    assert leading_zero["timestamp"] is not None

    one_to_many = conditions["10:2"]
    assert one_to_many["concept_ids"] == [101, 104]
    assert one_to_many["timestamp"] == leading_zero["timestamp"]

    acute_mi = conditions["11:1"]
    inferior = conditions["11:2"]
    assert 102 in acute_mi["targets"]
    assert 102 in inferior["targets"]
    assert acute_mi["diagnosis_id"] != inferior["diagnosis_id"]
    assert acute_mi["timestamp"] != leading_zero["timestamp"]

    unmatched = conditions["99:1"]
    assert unmatched["hadm_id"] == "99"
    assert unmatched["icd_code"] == "E8809"
    assert unmatched["timestamp"] is None
    assert unmatched["concept_ids"] == [104]

    conflict = conditions["10:5"]
    assert conflict["conflict"] is True
    assert conflict["icd_code"] is None
    assert {source["icd_code"] for source in conflict["sources"]} == {
        "04119",
        "I214",
    }
    assert conflict["concept_ids"] == [100, 102]

    assert all(DISEASE_CONCEPT_ID not in row["targets"] for row in conditions.values())
    assert 102 in history["repeated_concepts"]

    def diagnoses_reaching(concept_id):
        return {
            row["diagnosis_id"]
            for row in conditions.values()
            if concept_id in row["targets"]
        }

    # I214, 41071, and the conflicting copy. The duplicate 04119 does not
    # add a second diagnosis, and I2510's two concepts do not count 100 twice.
    assert diagnoses_reaching(102) == {"11:1", "11:2", "10:5"}
    assert diagnoses_reaching(100) == set(conditions)


# 04119 maps to both children. Each child reaches ancestor A.
_ANCESTOR = 200
_LEFT = 201
_RIGHT = 202


def _write_one_to_many_athena(root: Path, write_tsv) -> None:
    """ICD9CM 041.19 maps to two concepts, and both sit under ancestor A."""
    root.mkdir(parents=True, exist_ok=True)
    write_tsv(
        root / "CONCEPT.csv",
        [
            "concept_id",
            "concept_name",
            "domain_id",
            "vocabulary_id",
            "concept_class_id",
            "standard_concept",
            "concept_code",
            "invalid_reason",
        ],
        [
            (DISEASE_CONCEPT_ID, "Disease", "Condition", "SNOMED", "Clinical Finding", "S", "64572001", ""),
            (_ANCESTOR, "Ancestor A", "Condition", "SNOMED", "Clinical Finding", "S", "A", ""),
            (_LEFT, "Left child", "Condition", "SNOMED", "Clinical Finding", "S", "L", ""),
            (_RIGHT, "Right child", "Condition", "SNOMED", "Clinical Finding", "S", "R", ""),
            (8005, "Streptococcus", "Condition", "ICD9CM", "ICD9 code", "", "041.19", ""),
            (8006, "Hypertension", "Condition", "ICD9CM", "ICD9 code", "", "401.9", ""),
        ],
    )
    write_tsv(
        root / "CONCEPT_RELATIONSHIP.csv",
        ["concept_id_1", "concept_id_2", "relationship_id", "invalid_reason"],
        [
            (_ANCESTOR, DISEASE_CONCEPT_ID, "Is a", ""),
            (_LEFT, _ANCESTOR, "Is a", ""),
            (_RIGHT, _ANCESTOR, "Is a", ""),
            (8005, _LEFT, "Maps to", ""),
            (8005, _RIGHT, "Maps to", ""),
            (8006, _LEFT, "Maps to", ""),
        ],
    )
    write_tsv(
        root / "CONCEPT_ANCESTOR.csv",
        [
            "ancestor_concept_id",
            "descendant_concept_id",
            "min_levels_of_separation",
            "max_levels_of_separation",
        ],
        [
            (DISEASE_CONCEPT_ID, DISEASE_CONCEPT_ID, 0, 0),
            (DISEASE_CONCEPT_ID, _ANCESTOR, 1, 1),
            (_ANCESTOR, _ANCESTOR, 0, 0),
            (DISEASE_CONCEPT_ID, _LEFT, 2, 2),
            (_ANCESTOR, _LEFT, 1, 1),
            (_LEFT, _LEFT, 0, 0),
            (DISEASE_CONCEPT_ID, _RIGHT, 2, 2),
            (_ANCESTOR, _RIGHT, 1, 1),
            (_RIGHT, _RIGHT, 0, 0),
        ],
    )


def _write_one_to_many_mimic(root: Path, second_diagnosis: bool) -> None:
    """One 04119 row, plus an unmatched second diagnosis when requested."""
    hosp = root / "hosp"
    icu = root / "icu"
    hosp.mkdir(parents=True)
    icu.mkdir(parents=True)
    (hosp / "patients.csv").write_text(
        "subject_id,gender,anchor_age,anchor_year,dod\n1,F,40,2150,\n",
        encoding="utf-8",
    )
    (hosp / "admissions.csv").write_text(
        "subject_id,hadm_id,admittime,dischtime\n"
        "1,10,2150-01-01 00:00:00,2150-01-05 00:00:00\n",
        encoding="utf-8",
    )
    diagnoses = "subject_id,hadm_id,seq_num,icd_code,icd_version\n1,10,1,04119,9\n"
    if second_diagnosis:
        diagnoses += "1,99,1,4019,9\n"
    (hosp / "diagnoses_icd.csv").write_text(diagnoses, encoding="utf-8")
    (icu / "icustays.csv").write_text(
        "subject_id,hadm_id,stay_id,intime,outtime\n"
        "1,10,500,2150-01-01 01:00:00,2150-01-02 01:00:00\n",
        encoding="utf-8",
    )


def _diagnosis_matrix(histories) -> SimpleNamespace:
    """Count distinct diagnosis rows per rolled-up concept.

    The diagonal is 1 when at least two diagnosis rows reach that concept.
    Two mapped targets inside one row still contribute once.
    """
    concepts = sorted(
        {
            int(concept)
            for history in histories
            for row in history["conditions"]
            for concept in row["targets"]
        }
    )
    index = {concept: position for position, concept in enumerate(concepts)}
    rows: list[int] = []
    for history in histories:
        counts: dict[int, int] = {}
        for condition in history["conditions"]:
            for concept in set(condition["targets"]):
                counts[concept] = counts.get(concept, 0) + 1
        for concept, count in counts.items():
            if count >= 2:
                rows.append(index[concept])
    matrix = torch.zeros((len(concepts), len(concepts)))
    for position in rows:
        matrix[position, position] = 1
    return SimpleNamespace(matrix=matrix, concept_ids=concepts)


@pytest.mark.parametrize("second_diagnosis", [False, True])
def test_mimic_rollup_counts_one_diagnosis_once(tmp_path, second_diagnosis):
    """04119 keeps its leading zero and both targets, and counts once.

    Both mapped concepts reach ancestor A. One source diagnosis leaves the
    diagonal at 0. A second diagnosis id, stored on an admission that is
    not in the admission table, sets the diagonal to 1 and stays in the
    history.
    """
    example = _example()
    mimic_root = tmp_path / "mimic"
    athena_root = tmp_path / "athena"
    _write_one_to_many_mimic(mimic_root, second_diagnosis)
    _write_one_to_many_athena(athena_root, example._write_tsv)

    vocabulary = OMOPVocabulary(athena_root)
    mapper = vocabulary.icd_mapper()
    assert mapper.map("04119", 9) == [_LEFT, _RIGHT]
    assert mapper.map("4119", 9) == []
    hierarchy = vocabulary.condition_hierarchy(max_depth=5, depth_measure="min")
    for concept_id in (_LEFT, _RIGHT):
        ancestors = {
            ancestor for ancestor, _distance in hierarchy.retained_ancestors(concept_id)
        }
        assert _ANCESTOR in ancestors

    cache_name = f"keep_rollup_{int(second_diagnosis)}"
    cache_dir = _cache_dir(tmp_path, cache_name)
    try:
        histories = _histories(mimic_root, vocabulary, cache_dir)
        assert len(histories) == 1
        conditions = {
            row["diagnosis_id"]: row for row in histories[0]["conditions"]
        }
        source = conditions["10:1"]
        assert source["icd_code"] == "04119"
        assert source["icd_version"] == 9
        assert source["concept_ids"] == [_LEFT, _RIGHT]
        assert source["targets"].count(_ANCESTOR) == 1
        if second_diagnosis:
            unmatched = conditions["99:1"]
            assert unmatched["hadm_id"] == "99"
            assert unmatched["icd_code"] == "4019"
            assert unmatched["timestamp"] is None
            assert _ANCESTOR in unmatched["targets"]
            assert set(conditions) == {"10:1", "99:1"}
        else:
            assert set(conditions) == {"10:1"}

        result = _diagnosis_matrix(histories)
        a_idx = result.concept_ids.index(_ANCESTOR)
        diagonal = result.matrix.to_dense()[a_idx, a_idx].item()
        assert diagonal == int(second_diagnosis)
    finally:
        if os.name == "nt":
            shutil.rmtree(REPO / ".pyhealth_cache" / cache_name, ignore_errors=True)
