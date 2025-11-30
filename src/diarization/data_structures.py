from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

NumpyVector = NDArray[np.float32]
LabeledSegmentType = Tuple[float, float, str]


@dataclass
class SpeechSegment:
    """
    A segment of detected speech.

    Parameters
    ----------
    start : float
        Start time of the segment in seconds.
    end : float
        End time of the segment in seconds.
    """

    start: float
    end: float


@dataclass
class SegmentEmbedding:
    """
    Container for a speech segment and its associated speaker embedding.

    Parameters
    ----------
    segment : SpeechSegment
        The source speech segment.
    embedding : np.ndarray
        The computed speaker embedding vector. Shape is (embedding_dim,).
    """

    segment: SpeechSegment
    embedding: NumpyVector


@dataclass
class LabeledSegment:
    """
    A speech segment with an assigned speaker label.

    Parameters
    ----------
    start : float
        Start time of the segment in seconds.
    end : float
        End time of the segment in seconds.
    speaker : str
        The assigned speaker identifier (e.g., "speaker_0").
    """

    start: float
    end: float
    speaker: str

    def to_dict(self) -> Dict[str, Union[float, str]]:
        """
        Convert this LabeledSegment object into a JSON-friendly dict,
        rounding timestamps to 3 decimals.
        """
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
        }

    @staticmethod
    def list_to_dict(
        segments: List["LabeledSegment"],
    ) -> list[dict[str, Union[float, str]]]:
        """
        Convert a list of LabeledSegment objects into a list of JSON-friendly dicts.
        """
        return [seg.to_dict() for seg in segments]

    @staticmethod
    def from_dict(data: Dict[str, LabeledSegmentType]) -> "LabeledSegment":
        """
        Load a LabeledSegment object from a JSON dict.
        """
        return LabeledSegment(**data)

    @staticmethod
    def list_from_dict(
        list_data: List[Dict[str, LabeledSegmentType]]
    ) -> List["LabeledSegment"]:
        """
        Convert a list of dicts into a list of LabeledSegment objects.
        """
        return [LabeledSegment.from_dict(d) for d in list_data]
