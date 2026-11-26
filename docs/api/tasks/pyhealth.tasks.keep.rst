.. Author: lee-leonardo (NetID: ll37)

pyhealth.tasks.keep
===================

Overview
--------

Builds the sparse patient co-occurrence matrix that KEEP
(`Elhussein et al., CHIL 2025 <https://arxiv.org/abs/2510.05049>`_, appendix
A.4) feeds to its embedding stage: complete OMOP patient histories are
rolled up to the retained ancestors of an
:class:`~pyhealth.processors.AncestorRollupProcessor`, each pair of
conditions that at least two distinct diagnoses establishes counts once per
patient, and the counts land in one sparse tensor whose axes carry the
retained hierarchy vocabulary. The matrix, the retained graph, and the
embeddings trained on them therefore share one concept index.

:func:`build_keep_omop_ancestor_rollup` builds a processor over the uncut
SNOMED Disease subtree using minimum closure depths for target eligibility.
Both workflows can share that processor.

The OMOP workflow passes ``OMOPDataset.iter_patients()`` to
:func:`build_keep_cooccurrence`. The MIMIC workflow runs
:class:`KeepConditionHistoryTask` through ``dataset.set_task(task)`` and
passes the resulting histories to ``task.build_cooccurrence()``. Both
workflows compute patient presence before calling the sparse matrix
builder once for the cohort.

The builders accept single-pass iterables of patients or histories.
``iter_patients()`` uses Polars to collect the unique patient IDs and
then loads one patient's events at a time. PyHealth's OMOP and MIMIC
readers cache normalized events, and ``set_task()`` caches MIMIC task
samples. Configure ``cache_dir`` when constructing the dataset. The
standalone OMOP example writes synthetic CSVs to exercise those readers.
If a notebook or ETL job already supplies patient objects or mapped
history records, pass the iterable directly. Counting needs no files.
The KEEP wrapper tracks unique patient IDs, and the
:ref:`sparse counter <patient-cooccurrence>` stores distinct observed pairs.

Configure MIMIC-III with the supplied ``mimic3_keep.yaml`` through
``config_path``, or MIMIC-IV with ``mimic4_keep.yaml`` through
``ehr_config_path``. Load ``diagnoses_icd`` in the reader; these opt-in
configurations include the admissions needed to attach encounter start
times. The task caches one raw history per patient and preserves every
standard Condition concept mapped from each diagnosis. Its cohort method
passes those IDs directly to roll-up.

Within a patient, MIMIC-III uses ``row_id`` and MIMIC-IV uses
``(hadm_id, seq_num, icd_code, icd_version)`` to identify a diagnosis.
Exact copies collapse. Distinct MIMIC-IV keys survive even if their ICD
codes map to the same concepts. A missing code or version keeps its
unmapped diagnostic. Treat the resulting ``diagnosis_id`` as an opaque
string. The ``keep_condition_history_v2`` task rebuilds histories in a
separate cache and can reuse normalized dataset events.

Run the two synthetic examples from the checkout.

.. code-block:: console

    python -m examples.mimic3_keep_cooccurrence_matrix
    python -m examples.mimic4_keep_cooccurrence_matrix

Each example prints mapped patient histories before computing the matrix.
The MIMIC-III example shows distinct rows for the same ICD-9 code. The
MIMIC-IV example shows distinct codes with the same admission and sequence.

.. code-block:: python

    from functools import partial

    from pyhealth.graph.cooccurrence import build_cooccurrence
    from pyhealth.tasks import KeepConditionHistoryTask

    task = KeepConditionHistoryTask(
        mimic_version=4, vocabulary=vocabulary, vocabulary_release=release
    )
    histories = dataset.set_task(task)
    result = task.build_cooccurrence(
        histories,
        rollup=processor,
        concept_ids=concept_ids,
        matrix_builder=partial(build_cooccurrence, device="cpu"),
    )

Use the same ``matrix_builder`` keyword for the OMOP workflow. Select
``device="cuda:0"`` for a CUDA tensor, or bind a callable through
``transform`` to return weights derived from counts.

Semantics
^^^^^^^^^

- **Diagnosis identity.** One ``get_events(event_type="condition_occurrence")``
  call per patient. The OMOP config must expose ``condition_occurrence_id``,
  ``condition_concept_id``, ``condition_source_value`` and
  ``visit_occurrence_id`` on the condition table (the default OMOP config
  omits the diagnosis id and the source value). Each diagnosis row is
  identified by its ``condition_occurrence_id``: identical copies collapse,
  copies with conflicting concept id, source value, timestamp or visit id
  are rejected, rows without a visit id are kept, and distinct rows that
  share a visit and a code stay distinct.
- **Mapping.** A nonzero ``condition_concept_id`` wins; otherwise the
  source value's ``ICD9CM:``/``ICD10CM:`` prefix selects the ICD version
  and the code text (leading zeros preserved) goes to the peer's
  :class:`~pyhealth.medcode.ICDToSNOMEDMapper`. The roll-up targets of all
  mapped concepts of one diagnosis are unioned, so one-to-many ICD mappings
  and multi-parent roll-ups contribute each target at most once per
  diagnosis. Under ``on_unmapped="error"`` the processor's
  :class:`~pyhealth.processors.UnmappedCodeError` propagates; under
  ``"empty"``, wholly unmapped events are counted once by reason:
  ``missing_source``, ``unsupported_source_format``, ``empty_icd_mapping``
  or ``empty_rollup``.
- **Presence and counting.** A target is present for a patient after two
  distinct diagnoses resolve to it; each unordered pair of present targets
  counts once per patient, off-diagonal entries are mirrored, and the
  diagonal holds the patient prevalence of each concept.
- **Axes.** ``concept_ids`` must be a sequence containing each retained
  concept id exactly once and nothing else — the matrix vocabulary is the
  processor's :attr:`~pyhealth.processors.AncestorRollupProcessor.retained_codes`.
  The supplied order labels both axes and should be the order used to index
  the retained graph and later embeddings. Isolated concepts and concepts
  absent from the cohort keep zero rows and columns; an empty retained
  vocabulary yields a ``(0, 0)`` matrix.

The paper describes whole histories, a two-occurrence presence rule,
multi-parent roll-up and a vocabulary shared with the Node2Vec graph; the
diagnosis-row identity, the post-roll-up thresholding, the prevalence
diagonal and the mirrored pair counts are this project's choices.

Occurrence-rule experiments
---------------------------

Both ``build_keep_cooccurrence`` and ``task.build_cooccurrence`` accept
``min_occurrences`` (a positive integer, default 2) and ``occurrence_unit``
(``"diagnosis"`` by default, or ``"admission"``). Thresholding always follows
ICD mapping and ancestor roll-up. Changing the threshold does not change
raw-history caches, patient membership, retained axes, or mapping diagnostics.

Diagnosis support counts distinct diagnosis identities; admission support unions
all mapped targets within each patient's admission before counting admissions.
The caller must supply admission-level ``visit_occurrence_id`` / ``visit_id``
values. A generic OMOP visit ID is not necessarily an admission ID. In admission
mode a mapped diagnosis without a usable ID raises ``ValueError``. Unmapped
rows provide no support. Diagnosis mode continues to allow missing visit IDs.

.. code-block:: python

    result = task.build_cooccurrence(
        histories, rollup=processor, concept_ids=concept_ids,
        min_occurrences=2, occurrence_unit="admission",
    )

``examples/keep_occurrence_rule_ablation.ipynb`` compares one, two, and three
diagnosis rows with two admissions on ten invented patient histories. Its helper
uses the public OMOP and mapped-MIMIC counters, an independent trace counter,
and literal expected presence/counts. It reuses the ancestor-rollup example's
fixed dense/direct graph; it does not establish a new baseline or evaluate
clinical data. Run the same assertions headlessly with a fresh output directory:

.. code-block:: console

    python -m examples.keep_occurrence_rule_ablation --output /tmp/keep-occurrence-new

Synthetic example
-----------------

A toy hierarchy with the OMOP shape: ``"1"`` is the root, ``"10"`` (A) and
``"11"`` (B) are retained at depth 1, ``"12"`` (Z) is retained but isolated
(it has a supplied root depth and no edges), and the deep concepts ``801``,
``802`` (multi-parent) and ``803`` roll up to the retained vocabulary. The
ICD mapper is the peer's real class over a small polars table where ``K1``
maps one-to-many to ``801`` and ``802``:

.. code-block:: python

    from datetime import datetime
    import polars as pl
    from pyhealth.data import Patient
    from pyhealth.graph import KnowledgeGraph
    from pyhealth.graph.cooccurrence import build_cooccurrence
    from pyhealth.tasks import build_keep_cooccurrence
    from pyhealth.medcode import ICDToSNOMEDMapper
    from pyhealth.processors import AncestorRollupProcessor

    kg = KnowledgeGraph(
        triples=[
            ("1", "is_a", "10"), ("1", "is_a", "11"),
            ("10", "is_a", "801"), ("10", "is_a", "802"),
            ("11", "is_a", "802"), ("11", "is_a", "803"),
        ],
        entity2id={"1": 0, "10": 1, "11": 2, "12": 3,
                   "801": 4, "802": 5, "803": 6},
        relation2id={"is_a": 0},
    )
    codes = ("1", "10", "11", "12", "801", "802", "803")
    processor = AncestorRollupProcessor(
        knowledge_graph=kg,
        domain_by_code={code: "Condition" for code in codes},
        hierarchy_relation="is_a", root="1",
        target_policy="all_retained", include_root=False,
        max_depth=1, on_unmapped="empty",
        root_depths={"1": 0, "10": 1, "11": 1, "12": 1,
                     "801": 2, "802": 2, "803": 2},
    )
    mapper = ICDToSNOMEDMapper(pl.DataFrame({
        "icd_version": [10, 10, 10, 10, 9],
        "icd_code": ["K1", "K1", "K2", "UNKN", "0010"],
        "source_code": ["K1", "K1", "K2", "UNKN", "001.0"],
        "source_concept_id": [9001, 9001, 9002, 9005, 9003],
        "concept_id": [801, 802, 801, 999, 803],
        "concept_name": ["Deep X", "Multi-parent", "Deep X", "Outside",
                         "Deep Y"],
        "target_domain_id": ["Condition"] * 5,
        "target_vocabulary_id": ["SNOMED"] * 5,
    }))

    def patient_with(rows):
        frame = pl.DataFrame({
            "patient_id": [rows[0][0]] * len(rows),
            "event_type": ["condition_occurrence"] * len(rows),
            "timestamp": [datetime(2020, 1, 1)] * len(rows),
            "_source_row": list(range(len(rows))),
            "condition_occurrence/condition_occurrence_id":
                [dx for _, dx, _, _, _ in rows],
            "condition_occurrence/condition_concept_id":
                [concept for _, _, concept, _, _ in rows],
            "condition_occurrence/condition_source_value":
                [source for _, _, _, source, _ in rows],
            "condition_occurrence/visit_occurrence_id":
                [visit for _, _, _, _, visit in rows],
        })
        return Patient(patient_id=rows[0][0], data_source=frame)

    # Patient 1: two different diagnoses across two visits, each resolving
    # to A and B (the first reaches A through two ICD targets); patient 2:
    # a single such diagnosis; patient 3: three distinct A diagnoses in one
    # visit.
    patients = [
        patient_with([
            ("1", "101", None, "ICD10CM:K1", "v1"),
            ("1", "102", "802", None, "v2"),
        ]),
        patient_with([("2", "201", None, "ICD10CM:K1", "v1")]),
        patient_with([
            ("3", "301", "801", None, "v9"),
            ("3", "302", "801", None, "v9"),
            ("3", "303", "801", None, "v9"),
        ]),
    ]

    result = build_keep_cooccurrence(
        patients, rollup=processor, icd_mapper=mapper,
        concept_ids=["11", "12", "10"],   # axes B, Z, A
        matrix_builder=lambda rows, *, concept_ids: build_cooccurrence(
            rows, concept_ids=concept_ids
        ),
    )
    result.matrix.to_dense().tolist()
    # [[1, 0, 1], [0, 0, 0], [1, 0, 2]]
    result.concept_ids
    # ('11', '12', '10')
    result.patients_processed, result.patients_with_present_targets
    # (3, 2)
    result.unique_diagnosis_events, result.mapped_events
    # (6, 6)
    dict(result.unmapped_events)
    # {'missing_source': 0, 'unsupported_source_format': 0,
    #  'empty_icd_mapping': 0, 'empty_rollup': 0}

The matrix shows the two rules the fixture pins down: patient 2's single
diagnosis cannot establish a condition by converging on it from several
mapped concepts, and patient 3's third qualifying diagnosis adds no extra
pair count. Prevalence sits on the diagonal (A for patients 1 and 3, B for
patient 1), the pair (A, B) is mirrored, and the isolated Z keeps a zero
row and column.

OMOP end-to-end
---------------

The runnable script
`examples/omop_keep_cooccurrence_matrix.py
<https://github.com/sunlabuiuc/PyHealth/blob/master/examples/omop_keep_cooccurrence_matrix.py>`_
shows the real path on synthetic inputs, without Athena or MIMIC
credentials: a minimal OMOP CSV cohort whose condition rows follow the peer
converter's conventions (one ``condition_occurrence_id`` per row,
``condition_concept_id`` 0 beside an ``ICD9CM:code``/``ICD10CM:code``
source value), a synthetic Athena bundle, and the KEEP profile —

.. code-block:: python

    from pyhealth.tasks import build_keep_cooccurrence, build_keep_omop_ancestor_rollup

    processor = build_keep_omop_ancestor_rollup(
        vocabulary,                       # pyhealth.medcode.OMOPVocabulary
        target_policy="all_retained",
        include_root=False,
        max_depth=5,                      # minimum closure levels from Disease
        path_source="closure",
        on_unmapped="empty",
    )
    result = build_keep_cooccurrence(
        dataset.iter_patients(),          # pyhealth.datasets.OMOPDataset
        rollup=processor,
        icd_mapper=vocabulary.icd_mapper(),
        concept_ids=sorted(processor.retained_codes),
    )

For later held-out evaluation, build the matrix using the histories
selected for embedding fitting, and keep the returned ``concept_ids``
order as the embedding vocabulary.

API Reference
-------------

.. py:module:: pyhealth.tasks.keep

.. autofunction:: pyhealth.tasks.keep.build_keep_omop_ancestor_rollup

.. autoclass:: pyhealth.tasks.keep.KeepConditionHistoryTask
    :members: build_cooccurrence
    :no-inherited-members:

.. autoclass:: pyhealth.tasks.keep.KeepCooccurrenceResult

.. autofunction:: pyhealth.tasks.keep.build_keep_cooccurrence
