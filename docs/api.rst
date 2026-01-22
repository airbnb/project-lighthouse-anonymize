API Reference
=============

This page contains the API reference for the Project Lighthouse Anonymize package.

Main Functions
--------------

These are the main functions exported by the top-level ``project_lighthouse_anonymize`` module.

.. autofunction:: project_lighthouse_anonymize.k_anonymize

.. autofunction:: project_lighthouse_anonymize.p_sensitize

.. autofunction:: project_lighthouse_anonymize.check_dq_meets_minimum_thresholds

.. autofunction:: project_lighthouse_anonymize.compute_score

.. autofunction:: project_lighthouse_anonymize.select_best_run

.. autofunction:: project_lighthouse_anonymize.prepare_gtrees

.. autodata:: project_lighthouse_anonymize.default_dq_metric_to_minimum_dq
   :annotation:

Constants
---------

.. automodule:: project_lighthouse_anonymize.constants
   :members:
   :undoc-members:
   :show-inheritance:

Data Quality Metrics
--------------------

Miscellaneous Metrics
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: project_lighthouse_anonymize.data_quality_metrics.misc
   :members:
   :undoc-members:
   :show-inheritance:

NMI (Normalized Mutual Information)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: project_lighthouse_anonymize.data_quality_metrics.nmi
   :members:
   :undoc-members:
   :show-inheritance:

Pearson Correlation
~~~~~~~~~~~~~~~~~~~

.. automodule:: project_lighthouse_anonymize.data_quality_metrics.pearson
   :members:
   :undoc-members:
   :show-inheritance:

RILM/ILM
~~~~~~~~

.. automodule:: project_lighthouse_anonymize.data_quality_metrics.rilm_ilm
   :members:
   :undoc-members:
   :show-inheritance:

Disclosure Risk Metrics
-----------------------

L-Diversity
~~~~~~~~~~~

.. automodule:: project_lighthouse_anonymize.disclosure_risk_metrics.l_diversity
   :members:
   :undoc-members:
   :show-inheritance:

P-Sensitive K-Anonymity
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: project_lighthouse_anonymize.disclosure_risk_metrics.p_sensitive_k_anonymity
   :members:
   :undoc-members:
   :show-inheritance:

Futures
-------

.. automodule:: project_lighthouse_anonymize.futures
   :members:
   :undoc-members:
   :show-inheritance:

Generalization Trees
--------------------

.. automodule:: project_lighthouse_anonymize.gtrees
   :members:
   :undoc-members:
   :show-inheritance:

Mondrian Algorithm
------------------

Core
~~~~

.. automodule:: project_lighthouse_anonymize.mondrian.core
   :members:
   :undoc-members:
   :show-inheritance:

Funnel Stats
~~~~~~~~~~~~

.. automodule:: project_lighthouse_anonymize.mondrian.funnel_stats
   :members:
   :undoc-members:
   :show-inheritance:

Implementation
~~~~~~~~~~~~~~

.. automodule:: project_lighthouse_anonymize.mondrian.implementation
   :members:
   :undoc-members:
   :show-inheritance:

K-Anonymity
~~~~~~~~~~~

.. automodule:: project_lighthouse_anonymize.mondrian.k_anonymity
   :members:
   :undoc-members:
   :show-inheritance:

Original
~~~~~~~~

.. automodule:: project_lighthouse_anonymize.mondrian.original
   :members:
   :undoc-members:
   :show-inheritance:

RILM
~~~~

.. automodule:: project_lighthouse_anonymize.mondrian.rilm
   :members:
   :undoc-members:
   :show-inheritance:

Tree
~~~~

.. automodule:: project_lighthouse_anonymize.mondrian.tree
   :members:
   :undoc-members:
   :show-inheritance:

P-Sensitize
-----------

.. automodule:: project_lighthouse_anonymize.p_sensitize
   :members:
   :undoc-members:
   :show-inheritance:

Pandas Utilities
----------------

.. automodule:: project_lighthouse_anonymize.pandas_utils
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: project_lighthouse_anonymize.utils
   :members:
   :undoc-members:
   :show-inheritance:

Wrappers
--------

Data Type Conversion
~~~~~~~~~~~~~~~~~~~~

.. automodule:: project_lighthouse_anonymize.wrappers.dtype_conversion
   :members:
   :undoc-members:
   :show-inheritance:

K-Anonymize Wrapper
~~~~~~~~~~~~~~~~~~~

.. automodule:: project_lighthouse_anonymize.wrappers.k_anonymize
   :members:
   :undoc-members:
   :show-inheritance:

P-Sensitize Wrapper
~~~~~~~~~~~~~~~~~~~

.. automodule:: project_lighthouse_anonymize.wrappers.p_sensitize
   :members:
   :undoc-members:
   :show-inheritance:

Shared Wrapper Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: project_lighthouse_anonymize.wrappers.shared
   :members:
   :undoc-members:
   :show-inheritance:
