pyhealth.processors.AncestorRollupProcessor
===========================================

Overview
--------

Processor that rolls observed concept codes up to their retained ancestors in
a condition hierarchy and projects the successful paths into one
:class:`torch_geometric.data.Data` graph per sample. Registered as
``"ancestor_rollup"`` in the processor registry.

It provides a configurable roll-up inspired by KEEP
(`Elhussein et al., CHIL 2025 <https://arxiv.org/abs/2510.05049>`_, appendix
A.1.1 and A.4): ICD diagnoses are converted to standard OMOP concepts, the
concepts are rolled up to eligible SNOMED ancestors within a depth limit of
the "Disease" root, and the paths that explain each roll-up become a graph.
The behaviour was rehearsed and validated against a standalone reference
before this port; the OMOP integration is built on the vocabulary reader of
`PR #2 <https://github.com/rishabshah/PyHealth/pull/2>`_ —
:class:`~pyhealth.medcode.OMOPVocabulary`,
:class:`~pyhealth.medcode.ConditionHierarchy` and
:class:`~pyhealth.medcode.ICDToSNOMEDMapper` — which this processor reuses
without modification.

Two entry points build the same internal roll-up index:

- the generic constructor takes a :class:`~pyhealth.graph.KnowledgeGraph`
  whose ``hierarchy_relation`` triples are supplied **parent -> child**, a
  domain mapping, and optionally authoritative root depths and closure pairs;
- :meth:`~pyhealth.processors.AncestorRollupProcessor.from_omop` converts
  the uncut SNOMED Disease subtree from
  :class:`~pyhealth.medcode.OMOPVocabulary` to a PyHealth KnowledgeGraph.
  Its direct edges point parent to child. Minimum ``CONCEPT_ANCESTOR``
  separations supply the root depths used for target eligibility.

Semantics
^^^^^^^^^

- **Depth cutoff.** ``max_depth`` (default 5, KEEP's setting) applies to
  *target eligibility* over the uncut hierarchy, not to hierarchy
  truncation: deep sources and intermediate concepts stay traversable, and
  the reader's own ``max_depth``/``restrict()`` remain separate knobs for
  graph-shaped consumers. The cutoff is computed before any truncation, and
  ``first_retained`` stopping is branch-local.
- **Target policies.** ``"first_retained"`` stops each upward branch at its
  first eligible ancestor; ``"all_retained"`` emits every distinct eligible
  strict ancestor. KEEP's actual choice is unresolved; the caller decides.
- **Failure reasons.** ``unknown_code``, ``outside_domain`` (non-Condition),
  ``unreachable`` (Condition concept outside the subtree), ``no_target``
  (reachable source with no eligible target on the selected edges); typed
  errors by default, empty results under ``on_unmapped="empty"``.
- **Path sources.** ``path_source="direct"`` uses the "Is a" edges;
  ``path_source="closure"`` additionally adds ``CONCEPT_ANCESTOR`` pairs at
  minimum separation one (direct provenance wins on overlap). Edges that
  exist only in the closure are flagged in the edge-aligned ``closure_only``
  mask of the projected graph. The selected edge set must be acyclic.
- **Occurrence counting.** :meth:`rollup_codes` returns one typed result per
  input occurrence, in input order; the projected graph merges nodes and
  edges. ICD conversion stays explicit through
  ``vocabulary.icd_mapper()``, so the caller keeps diagnosis event IDs and
  owns counting: converging targets within one event count once, distinct
  events stay distinct.

Synthetic example
-----------------

A deep source, the depth cutoff, both path modes, and a closure-only edge,
without any Athena download (closure pairs stand in for the
``CONCEPT_ANCESTOR`` rows):

.. code-block:: python

    from pyhealth.graph import KnowledgeGraph
    from pyhealth.processors import AncestorRollupProcessor

    # R is the root; A and B sit at depth 1, C at depth 2. M is a deep
    # concept whose only "Is a" parent was filtered out by the source
    # vocabulary, so it has no hierarchy edge at all (the "treats" row is
    # a canary that keeps M in the graph without making it a child of B).
    kg = KnowledgeGraph(triples=[
        ("R", "is_a", "A"), ("R", "is_a", "B"), ("A", "is_a", "C"),
        ("B", "treats", "M"),
    ])
    domains = {"R": "Condition", "A": "Condition", "B": "Condition",
               "C": "Condition", "M": "Condition"}

    # Authoritative root depths, e.g. OMOP CONCEPT_ANCESTOR separations:
    # M sits at depth 3 even though no "Is a" edge reaches it.
    root_depths = {"R": 0, "A": 1, "B": 1, "C": 2, "M": 3}

    base = dict(
        knowledge_graph=kg, domain_by_code=domains,
        hierarchy_relation="is_a", root="R",
        target_policy="first_retained", include_root=False,
        max_depth=2, on_unmapped="empty", root_depths=root_depths,
    )
    direct = AncestorRollupProcessor(**base)
    closure = AncestorRollupProcessor(
        path_source="closure", closure_pairs=[("A", "M")], **base
    )

    # With the cutoff at 2, C is retained (identity) while the deep M has
    # no eligible path on the direct edges alone:
    direct.rollup_codes(["C", "M"])
    # [RollupResult(source='C', targets=('C',), source_depth=2, reason='retained'),
    #  RollupResult(source='M', targets=(), source_depth=3, reason='no_target')]

    # The closure pair stitches M to its separation-one ancestor A:
    closure.rollup_codes(["M"])
    # [RollupResult(source='M', targets=('A',), source_depth=3, reason='rolled_up')]

    graph = closure.process(["M"])
    graph.node_ids       # tensor([0, 3])  -- global entity ids, here A and M
    graph.edge_index     # tensor([[0], [1]])  -- one edge, parent -> child
    graph.seed_mask      # tensor([False, True])  -- M was observed
    graph.target_mask    # tensor([True, False])  -- A is the target
    graph.closure_only   # tensor([True])  -- the edge exists only in the closure

    # Both modes keep the same authoritative depths; the fingerprint
    # records the configuration, the depth method and the edge provenance,
    # so task schemas distinguish them:
    direct.fingerprint == closure.fingerprint    # False

This is the synthetic shape of a real Athena discrepancy: some concepts have
a standard ancestor at ``CONCEPT_ANCESTOR`` separation one whose
``CONCEPT_RELATIONSHIP`` row is missing, because their only "Is a" route
runs through a valid-but-nonstandard intermediate that every filtered
hierarchy drops. For sources beyond the cutoff with no surviving direct
path, ``path_source="direct"`` reports ``no_target``. If the closure link
restores a path to an eligible ancestor, ``path_source="closure"`` returns
the target and flags the stitched edge.

Rolling up ICD diagnoses
------------------------

ICD conversion is explicit: the mapper runs first, the processor consumes
OMOP ``concept_id`` strings, and the caller keeps the diagnosis event IDs.
One ICD code can map to several concepts; each concept occurrence gets its
own result, and converging targets within one event count once:

.. code-block:: python

    from pyhealth.medcode import OMOPVocabulary
    from pyhealth.processors import AncestorRollupProcessor

    vocabulary = OMOPVocabulary("~/data/athena")
    processor = AncestorRollupProcessor.from_omop(
        vocabulary, target_policy="first_retained",
        include_root=False, max_depth=5,
    )
    mapper = vocabulary.icd_mapper()

    # A patient's diagnosis events; the event id is yours to keep.
    events = [
        ("visit-1:dx-0", 10, "I21.4"),
        ("visit-1:dx-1", 10, "I25.10"),   # maps to two concepts
        ("visit-2:dx-0", 10, "I21.A1"),
    ]

    for event_id, version, icd_code in events:
        results = [
            processor.rollup_codes([str(concept_id)])[0]
            for concept_id in mapper.map(icd_code, version=version)
        ]
        # one result per concept occurrence, in mapping order
        targets = {target for result in results for target in result.targets}
        # converging targets within the event count once; distinct events
        # stay distinct, because counting belongs to the task
        print(event_id, [result.source for result in results], sorted(targets))

Using pretrained node embeddings
--------------------------------

The graph stores string concept keys and assigns integer entity IDs.
``Data.node_ids`` carries those IDs in both entry points. Use
``processor.id2entity[node_id]`` to retrieve the concept key. For a generic
graph, pass the pretrained feature tensor as ``KnowledgeGraph.node_features``,
with rows aligned to ``entity2id``. For OMOP, pass the vectors by string
``concept_id`` and the factory aligns them to the graph.

.. code-block:: python

    import torch

    embedded = AncestorRollupProcessor.from_omop(
        vocabulary,
        target_policy="first_retained",
        include_root=True,
        node_features_by_code={"4274025": torch.tensor([0.1, 0.2])},
    )
    embedded.process(["4274025"]).x  # tensor([[0.1000, 0.2000]])

Use concept keys from your trained embedding artifact in place of the
illustrative vector. ``Data.x`` contains the rows for the projected nodes,
with zeros for graph nodes that have no supplied vector. Empty samples keep
the feature width for batching. Train KEEP embeddings separately using the
chosen graph vocabulary and patient-history co-occurrences.

API Reference
-------------

.. automodule:: pyhealth.processors.ancestor_rollup_processor
    :members:
    :show-inheritance:
