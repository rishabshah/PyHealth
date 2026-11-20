pyhealth.tasks.keep
===================

Build the KEEP ancestor roll-up from the OMOP vocabulary. The factory uses
the uncut SNOMED Disease subtree for traversal and minimum closure depths to
determine which concepts can be targets.

.. autofunction:: pyhealth.tasks.keep.build_keep_omop_ancestor_rollup
