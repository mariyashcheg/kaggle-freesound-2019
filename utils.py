import numpy as np
import os
import re


def get_new_model_path(path, suffix=''):
    numered_runs = []
    for x in os.listdir(path):
        r = re.match('(\d+)', x)
        if r:
            numered_runs.append((os.path.join(path, x), int(r.group())))

    numered_runs.sort(key=lambda t: t[1])
    if len(numered_runs) == 0:
        new_number = 0
    else:
        _, nums = zip(*numered_runs)
        new_number = nums[-1] + 1
    if suffix != '':
        suffix = '_' + suffix
    t = os.path.join(path, '{}{}'.format(new_number, suffix))
    os.mkdir(t)
    os.mkdir(os.path.join(t, 'eval'))
    return t


def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


class lwlrap_accumulator(object):
    """Accumulate batches of test samples into per-class and overall lwlrap."""

    def __init__(self):
        self.num_classes = 0
        self.total_num_samples = 0

    def accumulate_samples(self, batch_truth, batch_scores):
        """Cumulate a new batch of samples into the metric.

        Args:
          truth: np.array of (num_samples, num_classes) giving boolean
            ground-truth of presence of that class in that sample for this batch.
          scores: np.array of (num_samples, num_classes) giving the
            classifier-under-test's real-valued score for each class for each
            sample.
        """
        assert batch_scores.shape == batch_truth.shape
        num_samples, num_classes = batch_truth.shape
        if not self.num_classes:
            self.num_classes = num_classes
            self._per_class_cumulative_precision = np.zeros(self.num_classes)
            self._per_class_cumulative_count = np.zeros(self.num_classes,
                                                        dtype=np.int)
        assert num_classes == self.num_classes
        for truth, scores in zip(batch_truth, batch_scores):
            pos_class_indices, precision_at_hits = (
                _one_sample_positive_class_precisions(scores, truth))
            self._per_class_cumulative_precision[pos_class_indices] += (
                precision_at_hits)
            self._per_class_cumulative_count[pos_class_indices] += 1
        self.total_num_samples += num_samples

    def per_class_lwlrap(self):
        """Return a vector of the per-class lwlraps for the accumulated samples."""
        return (self._per_class_cumulative_precision /
                np.maximum(1, self._per_class_cumulative_count))

    def per_class_weight(self):
        """Return a normalized weight vector for the contributions of each class."""
        return (self._per_class_cumulative_count /
                float(np.sum(self._per_class_cumulative_count)))

    def overall_lwlrap(self):
        """Return the scalar overall lwlrap for cumulated samples."""
        return np.sum(self.per_class_lwlrap() * self.per_class_weight())
