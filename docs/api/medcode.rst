MedCode
===================================

Translating Between Medical Standards
-------------------------------------

Healthcare data comes in many different coding systems and standards. PyHealth's medical code mapping enables:

- Cross-system mapping (e.g., ICD9CM → CCSCM, NDC → ATC)
- Within-system ontology lookup (e.g., ancestors/descendants in ICD, ATC hierarchy)

Quick Examples
^^^^^^^^^^^^^^

.. code-block:: python

   from pyhealth.medcode import CrossMap, InnerMap

   # Cross-system mapping: ICD-9-CM → CCS
   icd9_to_ccs = CrossMap.load("ICD9CM", "CCSCM")
   print(icd9_to_ccs.map("82101"))  # example diagnosis code

   # Drug code mapping: NDC → ATC
   ndc_to_atc = CrossMap.load("NDC", "ATC")
   print(ndc_to_atc.map("00527051210"))

   # Within-system lookup: ICD-9-CM
   icd9cm = InnerMap.load("ICD9CM")
   print(icd9cm.lookup("428.0"))
   print(icd9cm.get_ancestors("428.0"))

   # Translating between ICD versions
   icd9_to_icd10 = CrossMap.load("ICD9CM", "ICD10CM")
   print(icd9_to_icd10.map("428.0"))          # ['I50.9']

   # Grouping ICD-10 into CCSR categories
   icd10_to_ccsr = CrossMap.load("ICD10CM", "CCSR")
   print(icd10_to_ccsr.map("I50.9"))          # ['CIR019']

   # Coarser groupings, useful when a cohort is small
   print(CrossMap.load("ICD9CM", "ICD9CHAPTER").map("428.0"))       # ['7']
   print(CrossMap.load("ICD10CM", "ICD10CHAPTER").map("E11.9"))     # ['E00-E89']
   print(CrossMap.load("ICD10CM", "ICD10BLOCK").map("A41.9"))       # ['A30-A49']

   # Binary chronic-condition flags
   print(CrossMap.load("ICD9CM", "CCI").map("428.0"))               # ['1']
   print(CrossMap.load("ICD9CM", "CCI").map("486"))                 # ['0']

   # Which source served the mapping
   print(icd10_to_ccsr.backend)                                     # 'icdmappings'

Downloads and cache
^^^^^^^^^^^^^^^^^^^

The first ``load()`` of a vocabulary downloads its mapping files over HTTPS from
PyHealth's resource bucket and caches them under ``~/.cache/pyhealth/medcode/``.
Later loads read the cache and need no network access.

Downloads are written to a temporary ``.part`` file and only moved into place when
complete, and they time out instead of hanging. An interrupted download therefore
never leaves a truncated file behind, and you can simply retry. To re-download a
vocabulary, pass ``refresh_cache=True``:

.. code-block:: python

   icd9cm = InnerMap.load("ICD9CM", refresh_cache=True)

We provide medical code mapping tools for (i) ontology mapping within one coding system and 
(ii) mapping the same concept cross different coding systems. 


.. autoclass:: pyhealth.medcode.InnerMap
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CrossMap
    :members:
    :undoc-members:
    :show-inheritance:

Diagnosis codes:
----------------------

.. autoclass:: pyhealth.medcode.ICD9CM
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.ICD10CM
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCSCM
    :members:
    :undoc-members:
    :show-inheritance:

Procedure codes:
----------------------

.. autoclass:: pyhealth.medcode.ICD9PROC
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.ICD10PROC
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCSPROC
    :members:
    :undoc-members:
    :show-inheritance:

Medication codes:
-----------------------

.. autoclass:: pyhealth.medcode.NDC
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.RxNorm
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.ATC
    :members:
    :undoc-members:
    :show-inheritance:




    

ICD translation and grouper vocabularies
-----------------------------------------

PyHealth hosts its own mapping tables and always prefers them. Pairs it has no
table for are served by the `icd-mappings <https://pypi.org/project/icd-mappings/>`_
package, whose data ships inside its wheel and therefore needs no network
access. ``CrossMap.backend`` reports which source was used, and the choice can
be forced with ``CrossMap.load(..., backend="pyhealth")`` or
``backend="icdmappings"``.

``ICD9CM->CCSCM`` is the one pair both sources can serve. It stays on PyHealth's
hosted table under ``backend="auto"`` so that existing pipelines are unaffected;
pass ``backend="icdmappings"`` explicitly to get the same mapping offline.

Supported pairs, by source:

================================  ==========================================
Pair                              Notes
================================  ==========================================
``ICD9CM``   -> ``ICD10CM``       NBER General Equivalence Mappings
``ICD10CM``  -> ``ICD9CM``        NBER General Equivalence Mappings
``ICD10CM``  -> ``CCSR``          530 refined categories, ICD-10 only
``ICD9CM``   -> ``CCI``           chronic condition indicator, ``"1"``/``"0"``
``ICD10CM``  -> ``CCIR``          refined chronic condition indicator
``ICD9CM``   -> ``ICD9CHAPTER``   19 chapters
``ICD10CM``  -> ``ICD10CHAPTER``  22 chapters, keyed by code range
``ICD10CM``  -> ``ICD10BLOCK``    226 blocks, keyed by code range
``ICD9CM``/``ICD10CM`` -> ``CCC``     pediatric complex chronic condition
``ICD9CM``/``ICD10CM`` -> ``CCCSUB``  its subcategory
================================  ==========================================

.. warning::

   ICD-9 <-> ICD-10 translation is a **primary-mapping approximation**, not the
   full many-to-many GEM relation. ``map()`` returns at most one target code,
   some codes have no mapping at all, and the translation does not round-trip
   (``428.0`` maps to ``I50.9``, which maps back to ``428.9``). Measured on the
   581 distinct ICD-9 codes in the MIMIC-III demo, 561 mapped and 17 distinct
   ICD-9 codes collapsed onto a shared ICD-10 code. Inspect
   ``CrossMap.unmapped_codes`` after a pass over a dataset to quantify the loss.

   For grouping mixed ICD-9/ICD-10 data into one feature space, prefer mapping
   both versions into a shared grouper (CCS or CCSR) rather than translating
   one version into the other.

.. autoclass:: pyhealth.medcode.FlatMap
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCSR
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCI
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCIR
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.ICD9CHAPTER
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.ICD10CHAPTER
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.ICD10BLOCK
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCC
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.CCCSUB
    :members:
    :undoc-members:
    :show-inheritance:

OMOP Vocabularies (Athena)
--------------------------

``OMOPVocabulary`` reads a local download of the OMOP standardized vocabularies
from `Athena <https://athena.ohdsi.org>`_. Athena vocabularies are licensed and
cannot be redistributed, so PyHealth does not download them: create an Athena
account, download a bundle that includes SNOMED, ICD9CM and ICD10CM, unzip it
outside your repository, and pass that folder to ``OMOPVocabulary``. Athena
writes tab-separated files; for vocabulary tables from another source, pass an
``OMOPVocabularyFormat`` (separator, quoting, encoding, file names).

It builds two objects:

- ``OMOPConditionHierarchy``: the SNOMED condition graph rooted at "Disease"
  (concept 4274025) and cut off at a chosen depth, as in KEEP
  (`Elhussein et al., CHIL 2025 <https://arxiv.org/abs/2510.05049>`_,
  Appendix A.1.1). By default a concept is kept when its shortest path to
  "Disease" in ``CONCEPT_ANCESTOR`` is at most five levels; every "Is a" link
  between kept concepts is an edge, so concepts with several parents keep all
  of them.
- ``ICDToSNOMEDMapper``: maps ICD-9-CM and ICD-10-CM diagnosis codes, written
  with or without dots (MIMIC omits them), to standard concepts through
  Athena's "Maps to" links. It accepts in-memory and lazy polars tables.

**Orphan rescue is an adapter policy, not part of KEEP.** Filtering to the
Condition domain leaves a few kept concepts whose only parents were dropped
(for example, a parent in the Observation domain). With
``connect_orphans=True`` (the default) each is attached to its nearest kept
ancestor, which may be several levels up. These edges are flagged ``rescued``
and exported under their own relation (``rescued_is_a`` /
``rescued_subsumes``) so they are never mistaken for direct "Is a" links. The
KEEP paper does not say how such concepts are handled; pass
``connect_orphans=False`` to leave them unattached.

.. code-block:: python

   import polars as pl
   from pyhealth.medcode import ICDToSNOMEDMapper, OMOPConditionHierarchy, OMOPVocabulary

   vocab = OMOPVocabulary("~/data/athena")

   # Build once from the multi-GB Athena files, then save for later runs
   hierarchy = vocab.condition_hierarchy(max_depth=5)
   hierarchy.save("~/.cache/pyhealth/omop/hierarchy_d5")
   hierarchy = OMOPConditionHierarchy.load("~/.cache/pyhealth/omop/hierarchy_d5")

   # Shallower graphs for a depth ablation, without re-reading Athena
   depth_3 = hierarchy.restrict(3)

   mapper = vocab.icd_mapper()
   concept_ids = mapper.map("I214", version=10)

   # Kept ancestors of any concept under "Disease", nearest first
   hierarchy.retained_ancestors(concept_ids[0])

   # Map a whole diagnoses table, e.g. MIMIC-IV hosp/diagnoses_icd; a
   # LazyFrame stays lazy, so this can sit inside a streaming pipeline
   diagnoses = pl.scan_csv("~/data/mimiciv/3.1/hosp/diagnoses_icd.csv.gz",
                           schema_overrides={"icd_code": pl.Utf8})
   mapped = mapper.map_frame(diagnoses)

Using the hierarchy as a ``KnowledgeGraph``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``to_knowledge_graph()`` returns a :class:`pyhealth.graph.KnowledgeGraph`
whose entities are OMOP ``concept_id`` values as strings. Every kept concept is
an entity, including any without edges. Depth, ancestor distances and rescue
flags stay on ``OMOPConditionHierarchy``; the graph only carries the edges and
their relation types.

Edge direction matters for ``GraphProcessor``: its k-hop extraction follows
edges *into* the seed codes. ``to_knowledge_graph()`` therefore defaults to
``direction="parent_to_child"`` (``(parent, "subsumes", child)``), so a
patient's codes reach their ancestors. ``to_triples()`` defaults to
``direction="child_to_parent"`` (``(child, "is_a", parent)``); from a child
seed, a graph built from those triples reaches only descendants.

The example below builds a three-concept hierarchy by hand; it needs
``torch-geometric`` for ``GraphProcessor``.

.. code-block:: python

   import polars as pl
   from pyhealth.medcode import OMOPConditionHierarchy
   from pyhealth.processors import GraphProcessor

   # Disease (4274025) -> Heart disease (100) -> Ischemic heart disease (101)
   hierarchy = OMOPConditionHierarchy(
       nodes=pl.DataFrame({
           "concept_id": [4274025, 100, 101],
           "concept_name": ["Disease", "Heart disease", "Ischemic heart disease"],
           "concept_code": ["64572001", "56265001", "414545008"],
           "min_depth": [0, 1, 2],
           "max_depth": [0, 1, 2],
           "depth": [0, 1, 2],
       }),
       edges=pl.DataFrame({
           "parent_id": [4274025, 100],
           "child_id": [100, 101],
           "rescued": [False, False],
       }),
       ancestors=pl.DataFrame({
           "concept_id": [100, 101, 101],
           "ancestor_id": [4274025, 100, 4274025],
           "distance": [1, 1, 2],
       }),
       root_concept_id=4274025,
       max_depth=5,
       depth_measure="min",
   )

   # Triples point child -> parent by default
   assert hierarchy.to_triples() == [("100", "is_a", "4274025"), ("101", "is_a", "100")]

   # The KnowledgeGraph points parent -> child, keyed by string concept ids
   kg = hierarchy.to_knowledge_graph()
   assert set(kg.entity2id) == {"4274025", "100", "101"}
   assert set(kg.relation2id) == {"subsumes", "rescued_subsumes"}

   # From the child seed, GraphProcessor reaches its ancestors
   graph = GraphProcessor(knowledge_graph=kg, num_hops=2).process(["101"])
   assert {kg.id2entity[i] for i in graph.node_ids.tolist()} == {"101", "100", "4274025"}

   # Metadata stays on the hierarchy
   assert hierarchy.depth(101) == 2
   assert hierarchy.retained_ancestors(101) == [(100, 1), (4274025, 2)]

.. autoclass:: pyhealth.medcode.OMOPVocabulary
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.OMOPVocabularyFormat
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.OMOPConditionHierarchy
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.medcode.ICDToSNOMEDMapper
    :members:
    :show-inheritance:

To roll conditions up to retained ancestors and project the paths into
per-sample graphs, use :func:`~pyhealth.tasks.keep.build_keep_omop_ancestor_rollup`.
