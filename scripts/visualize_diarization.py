import argparse
import json
from pathlib import Path
from typing import List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from diarization.data_structures import LabeledSegment
from diarization.pipeline import load_audio


def main(args: argparse.Namespace) -> None:
    """
    Load audio and diarization results, then generate and save a visualization plot.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing 'audio' and 'output_dir'.
    """
    waveform, sr = load_audio(args.audio)

    tgt_file_name = Path(args.audio).stem + "_diarization.json"
    diarization_file = Path(args.output_dir) / tgt_file_name
    with open(diarization_file, "r") as f:
        seg_list = json.load(f)

    segments = [
        LabeledSegment(
            start=s["start"],
            end=s["end"],
            speaker=s["speaker"],
        )
        for s in seg_list
    ]

    fig = plot_diarization_waveform(waveform, sr, segments)

    plt.tight_layout()
    fig_file = Path(args.audio).parent / (Path(args.audio).stem + "_viz.png")
    fig.savefig(fig_file)
    plt.show()
    print(f"Fig saved @ {fig_file}")


def plot_diarization_waveform(
        waveform: torch.Tensor,
        sr: int,
        segments: List[LabeledSegment],
        title: str = "Speaker Diarization",
        figsize: Tuple[float, float] = (16, 4)
) -> Figure:
    """
       Plot waveform + diarization segments.

       Parameters
       ----------
       waveform : Any
           The audio waveform. Can be a numpy array or a torch Tensor.
           If it's a 2D array, the first channel is used.
       sr : int
           Sample rate of the audio.
       segments : List[LabeledSegment]
           List of segments with speaker labels to visualize.
       title : str, optional
           The title of the plot. Default is "Speaker Diarization".
       figsize : Tuple[float, float], optional
           The size of the figure (width, height). Default is (16, 4).

       Returns
       -------
       matplotlib.figure.Figure
           The created matplotlib figure.
       """

    waveform = waveform.numpy()

    # Time axis
    duration = len(waveform) / sr
    time = np.linspace(0, duration, len(waveform))

    # Normalize waveform for plotting
    waveform_norm = waveform / np.max(np.abs(waveform))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, waveform_norm, color="lightgray", linewidth=0.8, label="Waveform")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.2, 1.2)

    # Assign colors per speaker
    import matplotlib.cm as cm
    speaker_ids = [seg.speaker for seg in segments]
    unique_speakers = sorted(set(speaker_ids))
    colors = {s: cm.tab10(i % 10) for i, s in enumerate(unique_speakers)}

    # Draw diarization bars
    for seg in segments:
        start, end, spk = seg.start, seg.end, seg.speaker
        ax.add_patch(
            Rectangle(
                (start, -1.1),
                end - start,
                2.2,
                color=colors[spk],
                alpha=0.3,
                label=spk  # assign speaker directly
            )
        )

    # Build legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, lbl in zip(handles, labels):
        if lbl not in unique:
            unique[lbl] = h

    ax.legend(unique.values(), unique.keys(), loc="upper right", title="Speakers")

    return fig
