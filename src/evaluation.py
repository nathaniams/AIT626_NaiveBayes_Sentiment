"""Evaluation metrics for binary sentiment classification (no scikit-learn)."""


def accuracy_score(y_true, y_pred):
    """Return fraction of correctly predicted labels."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Cannot evaluate an empty test set")
    return sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(y_true, y_pred):
    """Rows = true labels, columns = predicted labels; order [negative=0, positive=1]."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    cm = [[0, 0], [0, 0]]
    for actual, predicted in zip(y_true, y_pred):
        if actual not in (0, 1) or predicted not in (0, 1):
            raise ValueError("Only binary labels 0 (negative) and 1 (positive) are supported")
        cm[actual][predicted] += 1
    return cm


def class_metrics(cm, positive_label=1):
    """Precision, recall and F1 for one class (optional, beyond assignment minimum)."""
    if positive_label not in (0, 1):
        raise ValueError("positive_label must be 0 or 1")
    other = 1 - positive_label
    tp = cm[positive_label][positive_label]
    fp = cm[other][positive_label]
    fn = cm[positive_label][other]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp + fn}


def format_report(name, y_true, y_pred, model_summary=''):
    """Build the printable / loggable report for one experiment."""
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    neg = class_metrics(cm, positive_label=0)
    pos = class_metrics(cm, positive_label=1)
    lines = [
        f'Experiment: {name}',
        f'Accuracy: {acc:.4f} ({correct}/{len(y_true)})',
        'Confusion matrix (ROWS=true, COLUMNS=predicted):',
        '                   Pred Negative    Pred Positive',
        f'  True Negative        {cm[0][0]:>6}           {cm[0][1]:>6}',
        f'  True Positive        {cm[1][0]:>6}           {cm[1][1]:>6}',
        f'Negative: precision={neg["precision"]:.4f} recall={neg["recall"]:.4f} F1={neg["f1"]:.4f}',
        f'Positive: precision={pos["precision"]:.4f} recall={pos["recall"]:.4f} F1={pos["f1"]:.4f}',
    ]
    if model_summary:
        lines.extend(['Model:', model_summary])
    return '\n'.join(lines), acc, cm
