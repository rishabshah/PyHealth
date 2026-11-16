"""MIMIC-IV data in OMOP CDM form, loaded through PyHealth's OMOP dataset."""

from pathlib import Path

from pyhealth.data import Patient

from .omop import OMOPDataset

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "omop.yaml"

# OMOP uses 0 when a source code has no standard concept.
_UNMAPPED_CONCEPTS = {None, "", 0, "0"}


class MIMICOMOPDataset(OMOPDataset):
    """OMOP CDM tables produced from MIMIC-IV, as PyHealth patients.

    Person, visit, and death tables load through :class:`OMOPDataset`.
    ``condition_occurrence`` is added on top of those defaults so
    :class:`~pyhealth.patient_builder.PatientBuilder` can filter the cohort.
    Each condition event keeps its diagnosis id, concept id, source value,
    source concept id, status, dates, and visit id. ``get_condition_code`` returns the
    standard concept id when one is mapped, and the source code when
    ``condition_concept_id`` is 0.

    The CSV directory is produced separately; see ``patient_builder/convert.py``
    on the ``patient_builder`` branch.

    Examples:
        >>> from pyhealth.datasets import MIMICOMOPDataset
        >>> dataset = MIMICOMOPDataset(root="/path/to/mimic_iv_omop")  # doctest: +SKIP
        >>> patient = dataset.get_patient(dataset.unique_patient_ids[0])  # doctest: +SKIP
        >>> patient.get_events(event_type="condition_occurrence")  # doctest: +SKIP
        >>> dataset.get_visit_count(patient)  # doctest: +SKIP
        2
    """

    def __init__(
        self,
        root: str | Path,
        dev: bool = False,
        cache_dir: str | Path = ".pyhealth_cache",
    ) -> None:
        """Load MIMIC-IV OMOP tables from ``root``.

        Args:
            root: Directory of OMOP CSV files (``person.csv``,
                ``visit_occurrence.csv``, ``death.csv``,
                ``condition_occurrence.csv``).
            dev: When true, PyHealth keeps only the first 1000 patients.
            cache_dir: Directory for the processed event cache.
        """
        super().__init__(
            root=str(root),
            tables=["condition_occurrence"],
            dataset_name="mimic4_hosp_omop",
            config_path=str(CONFIG_PATH),
            dev=dev,
            cache_dir=str(cache_dir),
        )

    def get_condition_code(self, condition_event) -> str | None:
        """Return the condition concept id, or the source code if it is unmapped.

        Args:
            condition_event: A ``condition_occurrence`` event.

        Returns:
            The code as a string, or ``None`` when neither field is set.
        """
        concept = getattr(condition_event, "condition_concept_id", None)
        if concept not in _UNMAPPED_CONCEPTS:
            return str(concept)

        source = getattr(condition_event, "condition_source_value", None)
        if source not in (None, ""):
            return str(source)
        return None

    def get_visit_count(self, patient: Patient) -> int:
        """Return how many ``visit_occurrence`` events the patient has.

        Args:
            patient: A patient loaded from this dataset.

        Returns:
            The number of visits.
        """
        return len(patient.get_events(event_type="visit_occurrence"))
