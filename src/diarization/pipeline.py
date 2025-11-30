"""
Core pipeline orchestration.

This module brings together audio loading, VAD, embedding extraction, and
clustering to perform speaker diarization.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import librosa

from diarization.vad import energy_vad
from diarization.embeddings import ECAPAOnnxEmbeddingModel
from diarization.clustering import cluster_embeddings, assign_speakers
from diarization.data_structures import LabeledSegment, SpeechSegment

TARGET_SAMPLE_RATE = 16000
NUM_CHANNELS = 1

@dataclass
class DiarizationConfig:
    """
        Configuration for the speaker diarization pipeline.

        Parameters
        ----------
        model_path : str
            Path to the ONNX-exported speaker embedding model file.
        num_speakers : int
            The expected number of speakers to cluster.
        vad_energy_thresh : float, optional
            Energy threshold for Voice Activity Detection (0.0 to 1.0).
            Default is 0.05.
        """
    model_path: str
    num_speakers: int
    vad_energy_thresh: float

class DiarizationPipeline:
    """
        Main class for executing the speaker diarization workflow.

        This pipeline coordinates the following steps:
        1. Loading and preprocessing audio.
        2. Voice Activity Detection (VAD) to isolate speech segments.
        3. Extraction of speaker embeddings for each segment.
        4. Clustering embeddings to identify distinct speakers.
        5. Assigning speaker labels to the original segments.

        Parameters
        ----------
        config : DiarizationConfig
            Configuration object containing model paths and parameters.
        """

    def __init__(self, config: DiarizationConfig):
        self.config = config
        self.model = ECAPAOnnxEmbeddingModel(model_path=config.model_path)

    def run(self, audio_path: str) -> List[LabeledSegment]:
        """
                Execute the full diarization pipeline on an audio file.

                Parameters
                ----------
                audio_path : str
                    Path to the input WAV audio file.

                Returns
                -------
                List[LabeledSegment]
                    A list of temporal segments with assigned speaker labels.
                    Returns an empty list if no speech is detected.
                """
        waveform, sr = load_audio(audio_path)

        speech_segments: List[SpeechSegment] = energy_vad(
            waveform,
            sr,
            energy_thresh=self.config.vad_energy_thresh
        )

        if not speech_segments:
            return []

        seg_embs = self.model.embed_segments(waveform, sr, speech_segments)
        embeddings = [se.embedding for se in seg_embs]

        emb_matrix = np.stack(embeddings, axis=0).squeeze()

        labels = cluster_embeddings(
            emb_matrix,
            num_speakers=self.config.num_speakers,
        )

        labeled_segments = assign_speakers(speech_segments, labels)
        return labeled_segments


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """
    Load a mono WAV file and return (waveform, sample_rate).

    The audio is converted to mono (by averaging channels) and resampled
    to ``TARGET_SAMPLE_RATE`` if necessary.

    Parameters
    ----------
    path : str
        The path to the wave file to load.

    Returns
    -------
    waveform : torch.Tensor
        The tensor representation of the audio waveform (float32).
        Shape is (num_samples,).
    sample_rate : int
        The sample rate of the loaded audio. This matches ``TARGET_SAMPLE_RATE``.
    """
    audio, sr = librosa.load(path)
    if audio.ndim == 2:
        # stereo -> mono
        audio = np.mean(audio, axis=1)
    audio = audio.astype("float32")

    if sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    return torch.from_numpy(audio), sr
