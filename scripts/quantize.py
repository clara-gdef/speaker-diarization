import argparse

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, QuantType,
                                      quantize_static)


class DummyAudioReader(CalibrationDataReader):
    def __init__(self, num_batches=25, batch_size=1):
        self.num_batches = num_batches
        self.batch_size = batch_size

        # Pre-generate all calibration batches
        # Shape here is (batch_size, 100, 80)
        self._data = [
            {"fbank": np.random.randn(batch_size, 100, 80).astype("float32")}
            for _ in range(num_batches)
        ]
        self._iter = iter(self._data)

    def get_next(self):
        """
        Return the next batch as a dict {input_name: np.ndarray}
        or None when finished.
        """
        return next(self._iter, None)


def main(args):
    reader = DummyAudioReader()
    quantize_static(
        args.model_path,
        args.output_path,
        reader,
        weight_type=QuantType.QInt8,
    )

    print("INT8 quantization complete →", args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ECAPA-TDNN model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/embedding_model.onnx",
        help="Path to ECAPA-TDNN ONNX model.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/embedding_model_int8.onnx",
        help="Path to save quantized model.",
    )
    args = parser.parse_args()
    main(args)
