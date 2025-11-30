"""
Speaker embedding extraction utilities.

This module handles the loading of the ONNX-based speaker recognition model,
preprocessing of audio (Log-Mel Spectrograms), and extraction of embeddings
from speech segments.
"""

from typing import List

import librosa
import numpy as np
import onnxruntime as ort
import torch

from diarization.data_structures import (NumpyVector, SegmentEmbedding,
                                         SpeechSegment)


def compute_logmel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    win_length: int = 400,  # 25 ms @ 16kHz
    hop_length: int = 160,  # 10 ms @ 16kHz
    n_fft: int = 400,
) -> torch.Tensor:
    """
    Compute log-Mel filterbanks compatible with SpeechBrain ECAPA preprocessing.

    Parameters
    ----------
    waveform : torch.Tensor
        The input audio waveform.
    sample_rate : int, optional
        Sample rate of the audio. Default is 16000.
    n_mels : int, optional
        Number of Mel frequency bins. Default is 80.
    win_length : int, optional
        Window length for STFT. Default is 400 (25ms at 16kHz).
    hop_length : int, optional
        Hop length for STFT. Default is 160 (10ms at 16kHz).
    n_fft : int, optional
        FFT size. Default is 400.

    Returns
    -------
    torch.Tensor
        The log-Mel spectrogram. Shape is (n_mels, frames).
    """

    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        center=True,
        return_complex=True,
    )
    power = spec.abs().pow(2)  # (freq_bins, frames)

    mel_fb = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0.0,
        fmax=sample_rate / 2.0,
    )  # shape (n_mels, n_fft//2 + 1)

    mel_fb = torch.tensor(mel_fb, dtype=power.dtype)

    mel_spec = torch.matmul(mel_fb, power)  # (n_mels, frames)

    log_mel = torch.log(mel_spec + 1e-6)

    return log_mel


class ECAPAOnnxEmbeddingModel:
    """
    Wrap ONNX Runtime inference for the exported SpeechBrain ECAPA-TDNN model.

    The ONNX model is expected to take an input of shape
    (batch, n_mels=80, frames) and return an output of shape
    (batch, embedding_dim).

    Parameters
    ----------
    model_path : str
        Path to the ONNX model file.
    """

    def __init__(self, model_path: str):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _window_segment(
        self,
        waveform: torch.Tensor,
        sr: int,
        segment: SpeechSegment,
        window: float = 1.5,
        step: float = 0.75,
    ) -> List[torch.Tensor]:
        """
        Slice the waveform segment into overlapping chunks.

        Parameters
        ----------
        waveform : torch.Tensor
            The complete audio waveform.
        sr : int
            Sample rate of the audio.
        segment : SpeechSegment
            The specific segment to window.
        window : float, optional
            Length of the window in seconds. Default is 1.5.
        step : float, optional
            Step size (shift) between windows in seconds. Default is 0.75.

        Returns
        -------
        List[torch.Tensor]
            A list of waveform chunks.
        """
        start = int(segment.start * sr)
        end = int(segment.end * sr)
        seg_wave = waveform[start:end]

        window_len = int(window * sr)
        step_len = int(step * sr)

        if len(seg_wave) <= window_len:
            padded = torch.zeros(window_len, dtype=seg_wave.dtype)
            padded[: len(seg_wave)] = seg_wave
            return [padded]

        chunks = []
        for i in range(0, len(seg_wave) - window_len + 1, step_len):
            chunk = seg_wave[i : i + window_len]
            chunks.append(chunk)

        return chunks

    def embed_segment(
        self, waveform: torch.Tensor, sr: int, segment: SpeechSegment
    ) -> NumpyVector:
        """
        Compute the averaged speaker embedding for a single segment.

        The segment is split into windows, and an embedding is computed for each
        window. The final embedding is the mean of these window embeddings,
        normalized to unit length.

        Parameters
        ----------
        waveform : torch.Tensor
            The complete audio waveform.
        sr : int
            Sample rate of the audio.
        segment : SpeechSegment
            The speech segment to process.

        Returns
        -------
        np.ndarray
            The normalized speaker embedding vector.
        """
        windows = self._window_segment(waveform, sr, segment)
        all_embs = []

        for w in windows:
            # Compute SpeechBrain-style log-Mel features
            log_mel = compute_logmel_spectrogram(w, sr)  # torch (80, frames)

            # logmel: (80, T)
            x = log_mel.transpose(0, 1).unsqueeze(0).numpy().astype("float32")
            # # now: (1, T, 80)

            out = self.session.run([self.output_name], {self.input_name: x})[
                0
            ]  # shape (1, emb_dim)

            all_embs.append(out[0])

        emb = np.mean(all_embs, axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-10)

        return emb

    def embed_segments(
        self,
        waveform: torch.Tensor,
        sr: int,
        segments: List[SpeechSegment],
    ) -> List[SegmentEmbedding]:
        """
        Compute speaker embeddings for a list of speech segments.

        Parameters
        ----------
        waveform : torch.Tensor
            The complete audio waveform.
        sr : int
            Sample rate of the audio.
        segments : List[SpeechSegment]
            A list of speech segments.

        Returns
        -------
        List[SegmentEmbedding]
            A list containing the original segments and their computed embeddings.
        """
        result = []
        for seg in segments:
            emb = self.embed_segment(waveform, sr, seg)
            result.append(SegmentEmbedding(segment=seg, embedding=emb))

        return result
