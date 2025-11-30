"""
Command-line interface for the speaker diarization pipeline.

This module provides the entry point for running the diarization process
from the console. It parses arguments, initializes the pipeline, runs it
on the provided audio file, and outputs the results as JSON.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from diarization.data_structures import LabeledSegment
from diarization.pipeline import DiarizationConfig, DiarizationPipeline


def main(args: Any) -> None:
    """
    Execute the diarization pipeline with command-line arguments.

    Configures the pipeline based on the provided arguments, processes the audio,
    and prints or saves the resulting diarization in JSON format.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing:
        - audio: Path to input audio file.
        - model_path: Path to ONNX model.
        - num_speakers: Number of speakers to detect.
        - vad_thresh: Energy threshold for VAD.
        - output: Output file path or "-" for stdout.
    """
    config = DiarizationConfig(
        model_path=args.model_path,
        num_speakers=args.num_speakers,
        vad_energy_thresh=args.vad_thresh,
    )

    pipeline = DiarizationPipeline(config)

    labeled_segments = pipeline.run(args.audio)

    result_json = LabeledSegment.list_to_dict(labeled_segments)
    # Output to stdout
    if args.output == "-" or not args.output:
        print(json.dumps(result_json, indent=2))
        return

    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output filename
    tgt_file_name = Path(args.audio).stem + "_diarization.json"
    output_path = output_dir / tgt_file_name

    # Write JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    print(f"Output saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal speaker diarization pipeline using ECAPA ONNX."
    )

    parser.add_argument(
        "audio",
        type=str,
        help="Path to WAV file (mono, 16kHz preferred).",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/embedding_model.onnx",
        help="Path to ECAPA-TDNN ONNX model (exported from SpeechBrain).",
    )

    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Number of speakers to cluster.",
    )

    parser.add_argument(
        "--vad-thresh",
        type=float,
        default=0.1,
        help="Energy threshold for simple VAD (0-1).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/outputs",
        help="Output path for JSON diarization result (default: stdout).",
    )

    args = parser.parse_args()
    main(args)
