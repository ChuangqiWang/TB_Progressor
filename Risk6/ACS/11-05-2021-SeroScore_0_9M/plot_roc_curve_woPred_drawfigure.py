# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 23:38:39 2021

@author: Chuangqi
"""

from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import auc, RocCurveDisplay # , #plot_roc_curve, auc, RocCurveDisplay

def plot_roc_curve_woPred_drawfigure(y_pred, y, *, sample_weight=None,
                   drop_intermediate=True, response_method="auto",
                   name=None, ax=None, pos_label=None, **kwargs):
    """Plot Receiver operating characteristic (ROC) curve.
    Extra keyword arguments will be passed to matplotlib's `plot`.
    Read more in the :ref:`User Guide <visualizations>`.
    Parameters
    ----------
    estimator : estimator instance
        Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
        in which the last estimator is a classifier.
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.
    y : array-like of shape (n_samples,)
        Target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    drop_intermediate : boolean, default=True
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.
    response_method : {'predict_proba', 'decision_function', 'auto'} \
    default='auto'
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. If set to 'auto',
        :term:`predict_proba` is tried first and if it does not exist
        :term:`decision_function` is tried next.
    name : str, default=None
        Name of ROC Curve for labeling. If `None`, use the name of the
        estimator.
    ax : matplotlib axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is created.
    pos_label : str or int, default=None
        The class considered as the positive class when computing the roc auc
        metrics. By default, `estimators.classes_[1]` is considered
        as the positive class.
        .. versionadded:: 0.24
    Returns
    -------
    display : :class:`~sklearn.metrics.RocCurveDisplay`
        Object that stores computed values.
    See Also
    --------
    roc_curve : Compute Receiver operating characteristic (ROC) curve.
    RocCurveDisplay : ROC Curve visualization.
    roc_auc_score : Compute the area under the ROC curve.
    Examples
    --------
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> from sklearn import datasets, metrics, model_selection, svm
    >>> X, y = datasets.make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = model_selection.train_test_split(
    ...     X, y, random_state=0)
    >>> clf = svm.SVC(random_state=0)
    >>> clf.fit(X_train, y_train)
    SVC(random_state=0)
    >>> metrics.plot_roc_curve(clf, X_test, y_test)  # doctest: +SKIP
    >>> plt.show()                                   # doctest: +SKIP
    """

    fpr, tpr, _ = roc_curve(y, y_pred, pos_label=pos_label,
                            sample_weight=sample_weight,
                            drop_intermediate=drop_intermediate)
    roc_auc = auc(fpr, tpr)

    viz = RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        estimator_name=name,
        pos_label=pos_label
    )

    return viz.plot(ax=ax, name=name, **kwargs)                                  # doctest: +SKIP
