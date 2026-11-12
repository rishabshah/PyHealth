"""Convert MIMIC-IV hosp CSVs into OMOP CDM tables."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from mimic_omop.concepts import (
    EHR,
    GENDER_CONCEPT,
    INPATIENT_VISIT,
    LOCAL_CONCEPT_START,
    NO_MATCHING_CONCEPT,
    RACE_ETHNICITY,
    diagnosis_vocabulary,
    procedure_vocabulary,
)

PATIENT_SCHEMA = {
    "subject_id": pl.Int64,
    "gender": pl.Utf8,
    "anchor_age": pl.Int32,
    "anchor_year": pl.Int32,
    "anchor_year_group": pl.Utf8,
    "dod": pl.Utf8,
}

ADMISSION_SCHEMA = {
    "subject_id": pl.Int64,
    "hadm_id": pl.Int64,
    "admittime": pl.Utf8,
    "dischtime": pl.Utf8,
    "deathtime": pl.Utf8,
    "admission_type": pl.Utf8,
    "admit_provider_id": pl.Utf8,
    "admission_location": pl.Utf8,
    "discharge_location": pl.Utf8,
    "insurance": pl.Utf8,
    "language": pl.Utf8,
    "marital_status": pl.Utf8,
    "race": pl.Utf8,
    "edregtime": pl.Utf8,
    "edouttime": pl.Utf8,
    "hospital_expire_flag": pl.Int32,
}

DIAGNOSIS_SCHEMA = {
    "subject_id": pl.Int64,
    "hadm_id": pl.Int64,
    "seq_num": pl.Int32,
    "icd_code": pl.Utf8,
    "icd_version": pl.Int32,
}

PROCEDURE_SCHEMA = {
    "subject_id": pl.Int64,
    "hadm_id": pl.Int64,
    "seq_num": pl.Int32,
    "chartdate": pl.Utf8,
    "icd_code": pl.Utf8,
    "icd_version": pl.Int32,
}


def convert(
    hosp_dir: Path,
    out_dir: Path,
    limit_patients: int | None = None,
) -> dict:
    """Read hosp CSVs and write OMOP CDM CSVs.

    ``limit_patients`` keeps the first N ``subject_id`` values. Omit it to
    convert every patient in ``patients.csv``.
    """
    hosp_dir = Path(hosp_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patients = _scan(hosp_dir / "patients.csv", PATIENT_SCHEMA)
    admissions = _scan(hosp_dir / "admissions.csv", ADMISSION_SCHEMA)
    diagnoses = _scan(hosp_dir / "diagnoses_icd.csv", DIAGNOSIS_SCHEMA)
    procedures = _scan(hosp_dir / "procedures_icd.csv", PROCEDURE_SCHEMA)

    if limit_patients is not None:
        keep = patients.sort("subject_id").head(limit_patients).select("subject_id")
        patients = patients.join(keep, on="subject_id", how="semi")
        admissions = admissions.join(keep, on="subject_id", how="semi")
        diagnoses = diagnoses.join(keep, on="subject_id", how="semi")
        procedures = procedures.join(keep, on="subject_id", how="semi")

    patients = patients.collect()
    admissions = admissions.collect()
    diagnoses = diagnoses.collect()
    procedures = procedures.collect()

    known_patients = patients.select("subject_id")
    admissions = admissions.join(known_patients, on="subject_id", how="semi")

    person, unmapped_races = _person(patients, admissions)
    visit, end_before_start = _visit(admissions)
    death = _death(patients, admissions)
    observation_period = _observation_period(visit, death)
    condition, dropped_dx = _condition(diagnoses, visit)
    procedure, dropped_px = _procedure(procedures, visit)
    condition, procedure, concept = _attach_source_concepts(condition, procedure)

    _write(person, out_dir / "person.csv")
    _write(visit, out_dir / "visit_occurrence.csv")
    _write(death, out_dir / "death.csv")
    _write(observation_period, out_dir / "observation_period.csv")
    _write(condition, out_dir / "condition_occurrence.csv")
    _write(procedure, out_dir / "procedure_occurrence.csv")
    _write(concept, out_dir / "concept.csv")
    _write(_cdm_source(limit_patients), out_dir / "cdm_source.csv")

    summary = {
        "source": str(hosp_dir),
        "output": str(out_dir),
        "limit_patients": limit_patients,
        "person": person.height,
        "visit_occurrence": visit.height,
        "death": death.height,
        "observation_period": observation_period.height,
        "condition_occurrence": condition.height,
        "procedure_occurrence": procedure.height,
        "source_concepts": concept.height,
        "diagnoses_dropped_without_visit": dropped_dx,
        "procedures_dropped_without_visit": dropped_px,
        "visits_with_end_before_start": end_before_start,
        "unmapped_race_values": unmapped_races,
        "condition_concept_id": (
            "0. Standard SNOMED concepts need an Athena vocabulary. "
            "ICD codes are in condition_source_value."
        ),
        "birth_month_day": (
            "MIMIC-IV only provides a shifted birth year. "
            "month_of_birth and day_of_birth are 1."
        ),
    }
    (out_dir / "etl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _scan(path: Path, schema: dict[str, pl.DataType]) -> pl.LazyFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing MIMIC-IV file: {path}")
    return pl.scan_csv(path, schema_overrides=schema)


def _person(
    patients: pl.DataFrame, admissions: pl.DataFrame
) -> tuple[pl.DataFrame, list[str]]:
    race_rows = [
        {"race": race, "race_concept_id": race_id, "ethnicity_concept_id": ethnicity_id}
        for race, (race_id, ethnicity_id) in RACE_ETHNICITY.items()
    ]
    race_map = pl.DataFrame(race_rows)
    earliest_race = (
        admissions.filter(pl.col("race").is_not_null())
        .sort(["subject_id", "admittime"])
        .group_by("subject_id", maintain_order=True)
        .agg(pl.col("race").first())
    )
    unmapped = (
        earliest_race.join(race_map, on="race", how="anti")
        .get_column("race")
        .unique()
        .sort()
        .to_list()
    )
    gender_map = pl.DataFrame(
        {
            "gender": list(GENDER_CONCEPT),
            "gender_concept_id": list(GENDER_CONCEPT.values()),
        }
    )
    person = (
        patients.join(earliest_race, on="subject_id", how="left")
        .join(race_map, on="race", how="left")
        .join(gender_map, on="gender", how="left")
        .with_columns(
            pl.col("race_concept_id").fill_null(NO_MATCHING_CONCEPT),
            pl.col("ethnicity_concept_id").fill_null(NO_MATCHING_CONCEPT),
            pl.col("gender_concept_id").fill_null(NO_MATCHING_CONCEPT),
            (pl.col("anchor_year") - pl.col("anchor_age")).alias("year_of_birth"),
        )
        .select(
            pl.col("subject_id").alias("person_id"),
            "gender_concept_id",
            "year_of_birth",
            pl.lit(1).alias("month_of_birth"),
            pl.lit(1).alias("day_of_birth"),
            pl.format(
                "{}-01-01 00:00:00",
                pl.col("anchor_year") - pl.col("anchor_age"),
            ).alias("birth_datetime"),
            "race_concept_id",
            "ethnicity_concept_id",
            pl.col("subject_id").cast(pl.Utf8).alias("person_source_value"),
            pl.col("gender").alias("gender_source_value"),
            pl.lit(NO_MATCHING_CONCEPT).alias("gender_source_concept_id"),
            pl.col("race").alias("race_source_value"),
            pl.lit(NO_MATCHING_CONCEPT).alias("race_source_concept_id"),
            pl.col("race").alias("ethnicity_source_value"),
            pl.lit(NO_MATCHING_CONCEPT).alias("ethnicity_source_concept_id"),
        )
        .sort("person_id")
    )
    return person, unmapped


def _visit(admissions: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    timed = admissions.with_columns(
        pl.col("admittime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
        pl.col("dischtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    ).filter(pl.col("admittime").is_not_null())
    end_before_start = timed.filter(pl.col("dischtime") < pl.col("admittime")).height
    visit = (
        timed.with_columns(
            pl.when(
                pl.col("dischtime").is_null() | (pl.col("dischtime") < pl.col("admittime"))
            )
            .then(pl.col("admittime"))
            .otherwise(pl.col("dischtime"))
            .alias("visit_end")
        )
        .select(
            pl.col("hadm_id").alias("visit_occurrence_id"),
            pl.col("subject_id").alias("person_id"),
            pl.lit(INPATIENT_VISIT).alias("visit_concept_id"),
            pl.col("admittime").dt.strftime("%Y-%m-%d").alias("visit_start_date"),
            pl.col("admittime").dt.strftime("%Y-%m-%d %H:%M:%S").alias("visit_start_datetime"),
            pl.col("visit_end").dt.strftime("%Y-%m-%d").alias("visit_end_date"),
            pl.col("visit_end").dt.strftime("%Y-%m-%d %H:%M:%S").alias("visit_end_datetime"),
            pl.lit(EHR).alias("visit_type_concept_id"),
            pl.col("admission_type").alias("visit_source_value"),
            pl.lit(NO_MATCHING_CONCEPT).alias("visit_source_concept_id"),
            pl.col("admission_location").alias("admitted_from_source_value"),
            pl.lit(NO_MATCHING_CONCEPT).alias("admitted_from_concept_id"),
            pl.col("discharge_location").alias("discharged_to_source_value"),
            pl.lit(NO_MATCHING_CONCEPT).alias("discharged_to_concept_id"),
        )
        .sort(["person_id", "visit_start_datetime", "visit_occurrence_id"])
    )
    return visit, end_before_start


def _death(patients: pl.DataFrame, admissions: pl.DataFrame) -> pl.DataFrame:
    in_hospital = (
        admissions.with_columns(
            pl.col("deathtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
        )
        .filter(pl.col("deathtime").is_not_null())
        .group_by("subject_id")
        .agg(pl.col("deathtime").max())
    )
    death = (
        patients.join(in_hospital, on="subject_id", how="left")
        .with_columns(pl.col("dod").str.to_date("%Y-%m-%d", strict=False))
        .filter(pl.col("dod").is_not_null() | pl.col("deathtime").is_not_null())
        .with_columns(
            pl.coalesce([pl.col("dod"), pl.col("deathtime").dt.date()]).alias("death_date"),
        )
        .with_columns(
            pl.when(
                pl.col("deathtime").is_not_null()
                & (pl.col("deathtime").dt.date() == pl.col("death_date"))
            )
            .then(pl.col("deathtime"))
            .otherwise(pl.col("death_date").cast(pl.Datetime))
            .alias("death_datetime")
        )
        .select(
            pl.col("subject_id").alias("person_id"),
            pl.col("death_date").dt.strftime("%Y-%m-%d"),
            pl.col("death_datetime").dt.strftime("%Y-%m-%d %H:%M:%S"),
            pl.lit(EHR).alias("death_type_concept_id"),
        )
        .sort("person_id")
    )
    return death


def _observation_period(visit: pl.DataFrame, death: pl.DataFrame) -> pl.DataFrame:
    spans = visit.group_by("person_id").agg(
        pl.col("visit_start_date").min().alias("observation_period_start_date"),
        pl.col("visit_end_date").max().alias("visit_end"),
    )
    death_dates = death.select("person_id", pl.col("death_date").alias("died"))
    return (
        spans.join(death_dates, on="person_id", how="left")
        .with_columns(
            pl.when(pl.col("died").is_null() | (pl.col("died") <= pl.col("visit_end")))
            .then(pl.col("visit_end"))
            .otherwise(pl.col("died"))
            .alias("observation_period_end_date")
        )
        .with_row_index("observation_period_id", offset=1)
        .select(
            "observation_period_id",
            "person_id",
            "observation_period_start_date",
            "observation_period_end_date",
            pl.lit(EHR).alias("period_type_concept_id"),
        )
        .sort("person_id")
    )


def _condition(diagnoses: pl.DataFrame, visit: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    coded = _coded_events(diagnoses, diagnosis_vocabulary)
    linked = coded.join(
        visit.select(
            "visit_occurrence_id",
            "person_id",
            "visit_start_date",
            "visit_start_datetime",
            "visit_end_date",
            "visit_end_datetime",
        ),
        left_on=["hadm_id", "subject_id"],
        right_on=["visit_occurrence_id", "person_id"],
        how="inner",
    )
    dropped = coded.height - linked.height
    condition = (
        linked.sort(["subject_id", "hadm_id", "seq_num"])
        .with_row_index("condition_occurrence_id", offset=1)
        .select(
            "condition_occurrence_id",
            pl.col("subject_id").alias("person_id"),
            pl.lit(NO_MATCHING_CONCEPT).alias("condition_concept_id"),
            pl.col("visit_start_date").alias("condition_start_date"),
            pl.col("visit_start_datetime").alias("condition_start_datetime"),
            pl.col("visit_end_date").alias("condition_end_date"),
            pl.col("visit_end_datetime").alias("condition_end_datetime"),
            pl.lit(EHR).alias("condition_type_concept_id"),
            pl.col("hadm_id").alias("visit_occurrence_id"),
            "condition_source_value",
            "source_vocabulary",
            "source_code",
            pl.col("seq_num").cast(pl.Utf8).alias("condition_status_source_value"),
        )
    )
    return condition, dropped


def _procedure(procedures: pl.DataFrame, visit: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    coded = _coded_events(procedures, procedure_vocabulary)
    visit_keys = visit.select(
        "visit_occurrence_id",
        "person_id",
        "visit_start_datetime",
    )
    linked = coded.join(
        visit_keys,
        left_on=["hadm_id", "subject_id"],
        right_on=["visit_occurrence_id", "person_id"],
        how="inner",
    )
    dropped = coded.height - linked.height
    procedure = (
        linked.with_columns(
            pl.coalesce(
                [
                    pl.col("chartdate").str.to_datetime("%Y-%m-%d", strict=False),
                    pl.col("visit_start_datetime").str.to_datetime(
                        "%Y-%m-%d %H:%M:%S", strict=False
                    ),
                ]
            ).alias("procedure_at")
        )
        .sort(["subject_id", "hadm_id", "seq_num"])
        .with_row_index("procedure_occurrence_id", offset=1)
        .select(
            "procedure_occurrence_id",
            pl.col("subject_id").alias("person_id"),
            pl.lit(NO_MATCHING_CONCEPT).alias("procedure_concept_id"),
            pl.col("procedure_at").dt.strftime("%Y-%m-%d").alias("procedure_date"),
            pl.col("procedure_at").dt.strftime("%Y-%m-%d %H:%M:%S").alias("procedure_datetime"),
            pl.lit(EHR).alias("procedure_type_concept_id"),
            pl.col("hadm_id").alias("visit_occurrence_id"),
            pl.col("condition_source_value").alias("procedure_source_value"),
            "source_vocabulary",
            "source_code",
        )
    )
    return procedure, dropped


def _coded_events(frame: pl.DataFrame, vocabulary_for) -> pl.DataFrame:
    """Attach a vocabulary id and a ``VOCAB:code`` source value."""
    versions = frame.get_column("icd_version").unique().drop_nulls().to_list()
    labels = pl.DataFrame(
        {
            "icd_version": [int(version) for version in versions],
            "source_vocabulary": [vocabulary_for(int(version)) for version in versions],
        }
    ).cast({"icd_version": pl.Int32})
    return (
        frame.with_columns(pl.col("icd_code").str.strip_chars().str.to_uppercase())
        .filter(
            pl.col("icd_code").is_not_null()
            & (pl.col("icd_code") != "")
            & pl.col("icd_version").is_not_null()
        )
        .join(labels, on="icd_version", how="left")
        .with_columns(
            pl.format("{}:{}", pl.col("source_vocabulary"), pl.col("icd_code")).alias(
                "condition_source_value"
            ),
            pl.col("icd_code").alias("source_code"),
        )
    )


def _attach_source_concepts(
    condition: pl.DataFrame, procedure: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    codes = pl.concat(
        [
            condition.select(
                pl.col("source_vocabulary").alias("vocabulary_id"),
                pl.col("source_code").alias("concept_code"),
                pl.lit("Condition").alias("domain_id"),
            ),
            procedure.select(
                pl.col("source_vocabulary").alias("vocabulary_id"),
                pl.col("source_code").alias("concept_code"),
                pl.lit("Procedure").alias("domain_id"),
            ),
        ]
    ).unique()
    concept = codes.sort(["vocabulary_id", "concept_code"]).with_row_index(
        "concept_id", offset=LOCAL_CONCEPT_START
    )
    by_condition = concept.filter(pl.col("domain_id") == "Condition").select(
        "vocabulary_id",
        "concept_code",
        pl.col("concept_id").alias("condition_source_concept_id"),
    )
    by_procedure = concept.filter(pl.col("domain_id") == "Procedure").select(
        "vocabulary_id",
        "concept_code",
        pl.col("concept_id").alias("procedure_source_concept_id"),
    )
    condition = (
        condition.join(
            by_condition,
            left_on=["source_vocabulary", "source_code"],
            right_on=["vocabulary_id", "concept_code"],
            how="left",
        )
        .drop(["source_vocabulary", "source_code"])
    )
    procedure = (
        procedure.join(
            by_procedure,
            left_on=["source_vocabulary", "source_code"],
            right_on=["vocabulary_id", "concept_code"],
            how="left",
        )
        .drop(["source_vocabulary", "source_code"])
    )
    concept = concept.select(
        "concept_id",
        pl.col("concept_code").alias("concept_name"),
        "domain_id",
        "vocabulary_id",
        pl.col("vocabulary_id").alias("concept_class_id"),
        pl.lit(None, dtype=pl.Utf8).alias("standard_concept"),
        "concept_code",
        pl.lit("1970-01-01").alias("valid_start_date"),
        pl.lit("2099-12-31").alias("valid_end_date"),
        pl.lit(None, dtype=pl.Utf8).alias("invalid_reason"),
    )
    return condition, procedure, concept


def _cdm_source(limit_patients: int | None) -> pl.DataFrame:
    description = (
        "MIMIC-IV hosp patients, admissions, diagnoses_icd, and procedures_icd. "
        "Standard condition and procedure concepts are unmapped."
    )
    if limit_patients is not None:
        description += f" Limited to the first {limit_patients} patients."
    return pl.DataFrame(
        {
            "cdm_source_name": ["MIMIC-IV hosp"],
            "cdm_source_abbreviation": ["MIMIC4-HOSP"],
            "source_description": [description],
            "cdm_version": ["5.3"],
            "vocabulary_version": ["local ICD source concepts; standard clinical concepts 0"],
        }
    )


def _write(frame: pl.DataFrame, path: Path) -> None:
    frame.write_csv(path)
