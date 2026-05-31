"""RAGAS evaluation harness for the ScikitDocs starter (REQ-068, M11).

Mirrors ``project/src/evaluation/run_eval.py`` with two ScikitDocs-specific
adaptations:

* ``load_golden_set`` reads the starter's column shape
  (``question,expected_doc_ids,min_hits,ground_truth_answer,query_type,version_sensitive``)
  rather than the capstone's ``question,ground_truth,contexts`` shape.
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
