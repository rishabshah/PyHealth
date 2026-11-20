pyhealth.datasets.MIMICOMOPDataset
==================================

The notebook path starts from :class:`pyhealth.datasets.MIMIC4Dataset` and
:class:`pyhealth.tasks.KeepPatientHistory`. See
``examples/mimic_omop_patient_history.py``. This class is the interchange
reader for a CDM export that was already written to disk.

:class:`pyhealth.patient_builder.PatientBuilder` is an exact-code cohort
filter. One condition row equal to the requested code selects the patient.
``with_min_visits`` counts every visit, including visits that do not carry
that diagnosis. Both filters are semi-joins on the event table, so a
selected patient keeps every event. ``iter_patients`` loads one patient at
a time, and ``build`` collects that iterator into a list.

``condition_records`` returns every diagnosis, including a row with no visit
link. A mapped row carries the standard concept. An unmapped row carries the
ICD code and version for :class:`pyhealth.medcode.ICDToSNOMEDMapper`.
``concepts_with_repeated_diagnoses`` keeps one copy of a repeated diagnosis
id and raises when that id appears again with a different record. The check
runs before mapping. Each diagnosis then contributes a target once. Two
diagnoses that each reach an ancestor count twice.

The reader loads ``person``, ``visit_occurrence``, and ``death`` through
:class:`pyhealth.datasets.OMOPDataset` and adds ``condition_occurrence`` from
``mimic_omop.yaml``. That opt-in file adds ``condition_occurrence_id`` and
``condition_source_value`` to the shared columns (concept id, visit id, and
timestamps). It does not select ``condition_status_concept_id``. The shared
``omop.yaml`` is unchanged. Unmapped conditions (``condition_concept_id`` of
0) fall back to ``condition_source_value``.

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
