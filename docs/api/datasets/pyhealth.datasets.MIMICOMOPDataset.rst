pyhealth.datasets.MIMICOMOPDataset
==================================

:class:`pyhealth.patient_builder.PatientBuilder` is an exact-code cohort
filter. One condition row equal to the requested code selects the patient.
``with_min_visits`` counts every visit, including visits that do not carry
that diagnosis. ``build`` returns :class:`pyhealth.data.Patient` objects.

``condition_records`` returns every diagnosis, including a row with no visit
link. A mapped row carries the standard concept. An unmapped row carries the
ICD code and version for the mapper in PR #2.
``concepts_with_repeated_diagnoses`` unions the rolled-up targets inside one
diagnosis before applying the two-occurrence threshold. Two paths from one
diagnosis to an ancestor count once. Two diagnoses that each reach that
ancestor count twice.

The builder reads :class:`pyhealth.datasets.MIMICOMOPDataset`, which loads
``person``, ``visit_occurrence``, and ``death`` through
:class:`pyhealth.datasets.OMOPDataset` and adds ``condition_occurrence``.
Each diagnosis event keeps ``condition_occurrence_id``,
``condition_concept_id``, ``condition_source_value``,
``condition_source_concept_id``,
``condition_status_concept_id``, ``condition_status_source_value``, the start
and end dates, and ``visit_occurrence_id``. Unmapped conditions
(``condition_concept_id`` of 0) fall back to ``condition_source_value``.

A runnable synthetic case is ``examples/mimic_omop_patient_history.py``.
The CSV directory is produced by the conversion command on the
``patient_builder`` branch:
https://github.com/rishabshah/PyHealth/blob/patient_builder/patient_builder/convert.py

API Reference
-------------

.. autoclass:: pyhealth.datasets.MIMICOMOPDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.patient_builder.PatientBuilder
    :members:
    :undoc-members:
    :show-inheritance:
