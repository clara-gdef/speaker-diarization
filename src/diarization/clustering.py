"""
Clustering and speaker assignment utilities.

This module provides functions to cluster speaker embeddings (using KMeans)
and assign speaker labels to the original speech segments. It also handles
post-processing steps like merging consecutive segments belonging to the
same speaker.
"""
from typing import List

import numpy as np
from sklearn.cluster import KMeans

from diarization.data_structures import SpeechSegment, LabeledSegment


def cluster_embeddings(
        embeddings: np.ndarray,
        num_speakers: int,
) -> np.ndarray:
    """
    Perform KMeans clustering on speaker embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        The matrix of speaker embeddings. Shape is (num_segments, embedding_dim).
    num_speakers : int
        The target number of clusters (speakers). If the number of embeddings
        is less than this value, the number of clusters is adjusted to match
        the number of embeddings.

    Returns
    -------
    np.ndarray
        An array of integer cluster labels corresponding to each input embedding.
    """
    if embeddings.shape[0] < num_speakers:
        num_speakers = embeddings.shape[0]
    kmeans = KMeans(n_clusters=num_speakers, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    return labels


def assign_speakers(
        segments: List[SpeechSegment],
        labels: np.ndarray,
) -> List[LabeledSegment]:
    """
        Assign speaker labels to speech segments.

        Converts raw speech segments and integer cluster labels into
        ``LabeledSegment`` objects with string identifiers (e.g., "speaker_0").
        Also merges consecutive segments from the same speaker.

        Parameters
        ----------
        segments : List[SpeechSegment]
            The list of original speech segments.
        labels : np.ndarray
            The array of integer labels corresponding to each segment.

        Returns
        -------
        List[LabeledSegment]
            A list of labeled segments, merged where appropriate.
        """
    labeled: List[LabeledSegment] = []
    for seg, lab in zip(segments, labels):
        labeled.append(
            LabeledSegment(
                start=seg.start,
                end=seg.end,
                speaker=f"speaker_{lab}",
            )
        )
    return merge_consecutive(labeled)


def merge_consecutive(segments: List[LabeledSegment]) -> List[LabeledSegment]:
    """
    Merge consecutive segments with the same speaker label.

    Segments are sorted by start time. If two adjacent segments belong to the
    same speaker and are close in time (within 0.05s), they are merged into a
    single segment.

    Parameters
    ----------
    segments : List[LabeledSegment]
        The list of labeled segments to process.

    Returns
    -------
    List[LabeledSegment]
        The list of merged labeled segments.
    """
    if not segments:
        return []

    segments = sorted(segments, key=lambda s: s.start)
    merged: List[LabeledSegment] = []
    current = segments[0]

    for seg in segments[1:]:
        if seg.speaker == current.speaker and seg.start <= current.end + 0.05:
            current.end = max(current.end, seg.end)
        else:
            merged.append(current)
            current = seg

    merged.append(current)
    return merged
