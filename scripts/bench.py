import argparse
import os
import time

import onnxruntime as ort
import psutil
from quantize import DummyAudioReader


def benchmark(model_path, dummy_reader):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Warm-up
    for _ in range(5):
        sess.run(None, dummy_reader.get_next())

    # Measure latency
    t0 = time.time()
    for _ in range(20):
        sess.run(None, dummy_reader.get_next())
    t1 = time.time()

    latency_ms = (t1 - t0) / 20 * 1000

    # CPU / Memory usage
    process = psutil.Process(os.getpid())
    cpu = process.cpu_percent(interval=0.1)
    mem = process.memory_info().rss / 1e6  # MB

    return latency_ms, cpu, mem


def main(args):
    models = {
        "FP32": args.model_fp32,
        "INT8": args.model_int8,
    }

    for name, path in models.items():
        lat, cpu, mem = benchmark(path, DummyAudioReader())
        print(f"--- {name} ---")
        print(f"Latency: {lat:.2f} ms")
        print(f"CPU: {cpu:.1f}%")
        print(f"Memory: {mem:.1f} MB\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ECAPA-TDNN models.")
    parser.add_argument(
        "--model_fp32",
        type=str,
        default="models/embedding_model.onnx",
        help="Path to FP32 ECAPA-TDNN ONNX model.",
    )
    parser.add_argument(
        "--model_int8",
        type=str,
        default="models/embedding_model_int8.onnx",
        help="Path to INT8 ECAPA-TDNN ONNX model.",
    )
    args = parser.parse_args()
    main(args)
