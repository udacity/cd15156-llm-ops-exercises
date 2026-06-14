.. _cross-validation:

=================================================
Cross-validation: evaluating estimator performance
=================================================

.. currentmodule:: sklearn.model_selection

Cross-validation is a procedure for evaluating an estimator's
generalisation performance by repeatedly partitioning the data into
training and validation folds. See :ref:`model-evaluation` for the
companion guide on scoring metrics.

The default cross-validator in scikit-learn is :class:`KFold` with
``n_splits=5``. For classification tasks consider :class:`StratifiedKFold`
so that each fold preserves the per-class frequency.

The basic K-Fold idiom
======================

Use :func:`cross_val_score` to evaluate a model with K-fold
cross-validation in a single call.

.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X, y = load_iris(return_X_y=True)
    scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
    print(scores.mean())

The returned :class:`numpy.ndarray` has one score per fold. Wrap with
:func:`numpy.mean` for a point estimate or with
:func:`numpy.std` for the spread.

Stratification considerations
=============================

For imbalanced classification, :class:`StratifiedKFold` preserves the
class proportions in each fold. This typically reduces variance in the
fold-level scores at the cost of slightly biased estimates when the
class ratio is extreme.

.. seealso::

    :ref:`grid-search` for parameter selection.

Repeated cross-validation
-------------------------

For small datasets, :class:`RepeatedKFold` runs K-Fold multiple times
with different random seeds and averages the results. This gives a
more stable estimate but multiplies compute cost.

.. versionadded:: 0.18
    ``RepeatedKFold`` was introduced.

Caveats and pitfalls
====================

Cross-validation with time-series data requires :class:`TimeSeriesSplit`
rather than naïve K-Fold — the latter leaks future information into the
training set. See :doc:`auto_examples/model_selection/plot_cv_indices`
for an illustration.
