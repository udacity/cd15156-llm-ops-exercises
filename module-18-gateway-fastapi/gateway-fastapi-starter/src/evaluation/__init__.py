"""RAGAS evaluation harness for the ScikitDocs starter.

Two ScikitDocs-specific design points:

* ``load_golden_set`` reads the starter's column shape
  (``question,expected_doc_ids,min_hits,ground_truth_answer,query_type,version_sensitive``).
* A ``deprecated_apis`` sub-metric is layered on top of the four RAGAS
  metrics. It scores whether any scikit-learn symbol cited in the
  generated answer appears in
  ``src/evaluation/deprecated_apis.py::DEPRECATED_APIS`` — a
  faithfulness-style fact-checker scoped to library-API correctness.
"""

from src.evaluation.run_eval import (
    DEFAULT_METRICS,
    build_eval_dataset,
    evaluate_pipeline,
    load_golden_set,
    summarize,
)

__all__ = [
    "DEFAULT_METRICS",
    "build_eval_dataset",
    "evaluate_pipeline",
    "load_golden_set",
    "summarize",
]
