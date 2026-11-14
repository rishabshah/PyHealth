"""Filter an OMOP dataset down to a cohort of PyHealth patients."""

from pyhealth.data import Patient


class PatientBuilder:
    """Fluent cohort builder that returns native PyHealth :class:`Patient` objects.

    Filters narrow ``unique_patient_ids`` and can be chained. ``build`` loads
    each remaining id with ``dataset.get_patient``, so the result works with
    the rest of the PyHealth dataset API.

    Examples:
        >>> from pyhealth.patient_builder import PatientBuilder
        >>> patients = (
        ...     PatientBuilder(dataset)
        ...     .with_condition("201826")
        ...     .with_min_visits(2)
        ...     .build()
        ... )  # doctest: +SKIP
    """

    def __init__(self, dataset) -> None:
        """Start a cohort from every patient in ``dataset``.

        Args:
            dataset: An OMOP-backed dataset with ``unique_patient_ids``,
                ``get_patient``, ``get_condition_code``, and ``get_visit_count``.
                :class:`~pyhealth.datasets.MIMICOMOPDataset` provides these.
        """
        self.dataset = dataset
        self.patient_ids = list(dataset.unique_patient_ids)

    def with_condition(self, condition_code: str) -> "PatientBuilder":
        """Keep patients who have ``condition_code`` on any condition event.

        Args:
            condition_code: Concept id or source code, compared as a string.

        Returns:
            This builder, so another filter can be chained.
        """
        wanted = str(condition_code)
        filtered_ids = []
        for patient_id in self.patient_ids:
            patient = self.dataset.get_patient(patient_id)
            conditions = patient.get_events(event_type="condition_occurrence")
            for condition in conditions:
                if self.dataset.get_condition_code(condition) == wanted:
                    filtered_ids.append(patient_id)
                    break
        self.patient_ids = filtered_ids
        return self

    def with_min_visits(self, min_visits: int) -> "PatientBuilder":
        """Keep patients with at least ``min_visits`` visits.

        Args:
            min_visits: Minimum number of ``visit_occurrence`` events.

        Returns:
            This builder, so another filter can be chained.
        """
        filtered_ids = []
        for patient_id in self.patient_ids:
            patient = self.dataset.get_patient(patient_id)
            if self.dataset.get_visit_count(patient) >= min_visits:
                filtered_ids.append(patient_id)
        self.patient_ids = filtered_ids
        return self

    def build(self) -> list[Patient]:
        """Load the cohort as PyHealth patients.

        Returns:
            One :class:`~pyhealth.data.Patient` per remaining patient id.
        """
        return [self.dataset.get_patient(patient_id) for patient_id in self.patient_ids]
