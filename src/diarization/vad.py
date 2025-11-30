from typing import List

import ipdb
import torch

from .data_structures import SpeechSegment

def energy_vad(
        waveform: torch.Tensor,
        sample_rate: int,
        frame_size: float = 0.025,
        frame_shift: float = 0.01,
        energy_thresh: float = 0.05,
        min_speech_duration: float = 0.0,
) -> List[SpeechSegment]:
    """
   Perform simple energy-based Voice Activity Detection (VAD).

   Calculates frame-wise energy, normalizes it by the maximum energy in the
   clip, and thresholds it to detect speech regions.

   Parameters
   ----------
   waveform : torch.Tensor
       The input audio waveform (1D tensor).
   sample_rate : int
       The sample rate of the audio in Hz.
   frame_size : float, optional
       Duration of each frame in seconds. Default is 0.025.
   frame_shift : float, optional
       Shift between consecutive frames in seconds. Default is 0.01.
   energy_thresh : float, optional
       Normalized energy threshold (0.0 to 1.0) for speech detection.
       Frames with energy ratio >= this value are considered speech. Default is 0.5.
   min_speech_duration : float, optional
       Minimum required duration in seconds for a speech segment to be valid.
       Default is 0.3.

   Returns
   -------
   List[SpeechSegment]
       A list of detected speech segments with start and end timestamps.
   """

    assert waveform.ndim == 1
    waveform = waveform.float()

    frame_len = int(frame_size * sample_rate)
    hop_len = int(frame_shift * sample_rate)
    num_frames = max(1, (len(waveform) - frame_len) // hop_len + 1)

    energies = []
    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        frame = waveform[start:end]
        if len(frame) == 0:
            break
        energies.append(frame.pow(2).mean().item())

    if not energies:
        return []

    max_energy = max(energies)
    if max_energy == 0.0:
        return []

    norm_energies = [e / max_energy for e in energies]
    speech_flags = [e >= energy_thresh for e in norm_energies]

    segments: List[SpeechSegment] = []
    in_speech = False
    seg_start_frame = 0

    for i, is_speech in enumerate(speech_flags):
        if is_speech and not in_speech:
            in_speech = True
            seg_start_frame = i
        elif not is_speech and in_speech:
            in_speech = False
            seg_end_frame = i
            start_time = seg_start_frame * frame_shift
            end_time = seg_end_frame * frame_shift + frame_size
            if end_time - start_time >= min_speech_duration:
                segments.append(SpeechSegment(start_time, end_time))

    # close last segment if waveform ends with speech
    if in_speech:
        seg_end_frame = len(speech_flags)
        start_time = seg_start_frame * frame_shift
        end_time = seg_end_frame * frame_shift + frame_size
        if end_time - start_time >= min_speech_duration:
            segments.append(SpeechSegment(start_time, end_time))

    return segments
