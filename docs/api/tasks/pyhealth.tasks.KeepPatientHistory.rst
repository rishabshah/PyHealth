pyhealth.tasks.KeepPatientHistory
================================

One patient history for KEEP, built with ``dataset.set_task`` from
:class:`pyhealth.datasets.MIMIC4Dataset`. Pass the opt-in config
``pyhealth/datasets/configs/mimic4_keep_ehr.yaml`` so ``hadm_id``,
``seq_num``, ``icd_code``, and ``icd_version`` are on each diagnosis and a
diagnosis with no matching admission is kept. ICD lookup uses
:class:`pyhealth.medcode.OMOPVocabulary`.

.. code-block:: python

    from pyhealth.datasets import MIMIC4Dataset
    from pyhealth.medcode import OMOPVocabulary
    from pyhealth.tasks import KeepPatientHistory

    vocabulary = OMOPVocabulary("/path/to/athena")
    history_task = KeepPatientHistory(vocabulary)
    dataset = MIMIC4Dataset(
        ehr_root="/path/to/mimic-iv/2.2",
        ehr_tables=["diagnoses_icd"],
        ehr_config_path="pyhealth/datasets/configs/mimic4_keep_ehr.yaml",
    )
    histories = dataset.set_task(history_task, num_workers=1)
    histories[0]["conditions"]

A runnable synthetic case is ``examples/mimic_omop_patient_history.py``.
Reading an already converted CDM export stays on
:class:`pyhealth.datasets.MIMICOMOPDataset`.

.. autoclass:: pyhealth.tasks.keep_patient_history.KeepPatientHistory
    :members:
    :undoc-members:
    :show-inheritance:
