"""OMOP standardized vocabularies (Athena) for PyHealth.

Reads a local download of the OMOP standardized vocabularies from Athena
(https://athena.ohdsi.org) and provides:

* :class:`OMOPVocabulary`: the reader (``vocabulary.py``).
* :class:`OMOPConditionHierarchy`: the SNOMED condition graph rooted at
  "Disease" and cut off at a chosen depth, as used by KEEP (Elhussein et al.,
  CHIL 2025, appendix A.1.1) (``condition_hierarchy.py``).
* :class:`ICDToSNOMEDMapper`: maps ICD-9-CM and ICD-10-CM diagnosis codes, in
  the dot-free form MIMIC stores them, to standard OMOP concepts
  (``icd_mapper.py``).

Example:
    >>> from pyhealth.medcode import OMOPVocabulary
    >>> vocab = OMOPVocabulary("~/data/athena")  # doctest: +SKIP
    >>> hierarchy = vocab.condition_hierarchy(max_depth=5)  # doctest: +SKIP
    >>> mapper = vocab.icd_mapper()  # doctest: +SKIP
    >>> concept_ids = mapper.map("I214", version=10)  # doctest: +SKIP
    >>> # kept ancestors of each target, nearest first: [(ancestor_id, levels)]
    >>> hierarchy.retained_ancestors(concept_ids[0])  # doctest: +SKIP
"""

from .condition_hierarchy import RELATIONS, OMOPConditionHierarchy
from .icd_mapper import ICD_VERSION_TO_VOCABULARY, ICDToSNOMEDMapper, normalize_icd_code
from .vocabulary import DISEASE_CONCEPT_ID, OMOPVocabulary, OMOPVocabularyFormat

__all__ = [
    "DISEASE_CONCEPT_ID",
    "ICD_VERSION_TO_VOCABULARY",
    "ICDToSNOMEDMapper",
    "OMOPConditionHierarchy",
    "OMOPVocabulary",
    "OMOPVocabularyFormat",
    "RELATIONS",
    "normalize_icd_code",
]
