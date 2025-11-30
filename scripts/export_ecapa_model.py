import os
from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.speaker import SpeakerRecognition
from torch import nn
import argparse
import onnxruntime as ort

device = "cuda" if torch.cuda.is_available() else "cpu"


def export_ecapa_onnx(onnx_dir, onnx_name):
    print("📥 Chargement du modèle SpeechBrain ECAPA-TDNN...")
    sr = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    # SpeechBrain model
    ecapa: nn.Module = sr.mods.embedding_model

    ecapa.eval()

    output_path = Path(onnx_dir) / onnx_name
    print(f"📦 Export ONNX → {output_path}")

    # (batch, frames=T, n_mels=80,)
    dummy = torch.randn(1, 100, 80)
    torch.onnx.export(
        ecapa,
        dummy,
        output_path,
        input_names=["fbank"],
        output_names=["embedding"],
        dynamic_axes={
            "fbank": {1: "frames"},  # variable length time frames
        },
        opset_version=13
    )

    print(f"✅ Export finished, model exported at {output_path}")

    ## sanity check

    sess = ort.InferenceSession(output_path)
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name

    x = np.random.randn(1, 120, 80).astype("float32")
    y = sess.run([out], {inp: x})[0]
    assert y.shape == (1, 1, 192)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the speaker recognition model for the diarization pipeline."
    )
    parser.add_argument(
        "--onnx_dir",
        type=str,
        default="models/onnx"
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="embedding_model.ckpt"
    )
    parser.add_argument(
        "--onnx_name",
        type=str,
        default="embedding_model.onnx"
    )

    args = parser.parse_args()

    os.makedirs(args.onnx_dir, exist_ok=True)

    export_ecapa_onnx(args.onnx_dir,
                      args.onnx_name
                      )
