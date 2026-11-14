pyhealth.datasets.MIMICOMOPDataset
==================================

MIMIC-IV hospital data converted to the OMOP Common Data Model and loaded as
PyHealth patients. The dataset reads the standard OMOP CSV tables and adds
helpers for diagnosis codes and visit counts. Unmapped conditions
(``condition_concept_id`` of 0) fall back to ``condition_source_value``.

Cohorts are built with :class:`pyhealth.patient_builder.PatientBuilder`, which
returns the same :class:`pyhealth.data.Patient` objects as the rest of PyHealth.

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
