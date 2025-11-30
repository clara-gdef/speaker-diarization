# Minimal Speaker Diarization Pipeline

This project implements a lightweight yet functional speaker diarization pipeline. It takes an audio file as input and outputs a JSON file listing "who spoke when".

![data/short_mulan_viz.png](data/short_mulan_viz.png)

## How the Pipeline Works

The pipeline processes audio in five sequential steps:

      [ Input WAV ]
            │
            ▼
    +----------------+
    | Audio Loading  |  (16kHz Mono)
    +----------------+
            │
            ▼
    +----------------+
    |      VAD       |  (Energy-based)
    +----------------+
            │
            ▼
    +----------------+
    |   Embedding    |  (ECAPA-TDNN ONNX)
    +----------------+
            │
            ▼
    +----------------+
    |   Clustering   |  (K-Means)
    +----------------+
            │
            ▼
    +----------------+
    | Label Assign.  |  (Merge Segments)
    +----------------+
            │
            ▼
     [ JSON Output ]

1.  **Audio Loading**: The input WAV file is loaded, converted to mono, and resampled to 16kHz (the native rate for the embedding model).
2.  **Voice Activity Detection (VAD)**: An energy-based VAD scans the audio to identify segments containing speech, discarding silence and low-energy noise.
3.  **Embedding Extraction**:
    -   Detected speech segments are sliced into sliding windows (1.5s duration).
    -   Log-Mel spectrograms are computed for each window.
    -   A **SpeechBrain ECAPA-TDNN** model (exported to ONNX) computes a dense vector representation (embedding) for each window.
    -   Embeddings are averaged to produce a single vector per speech segment.
4.  **Clustering**: **K-Means** clustering is applied to the segment embeddings to group them by speaker identity. The number of speakers must be specified.
5.  **Label Assignment**: Consecutive segments belonging to the same cluster are merged to produce the final diarization timeline.

## How to Run It

### Prerequisites
Install the dependencies:

````bash 
pip install -e .
````

### Running the Diarization
Use the CLI script to process an audio file:

````bash 
python scripts/diarize.py data/short_mulan.wav
--model-path models/embedding_model.onnx
--num-speakers 2
--output data/outputs
````


### Visualization
You can generate a plot of the waveform with colored speaker segments:


````bash
bash python scripts/visualize_diarization.py data/short_mulan.wav --output_dir data/outputs
````


## Design Decisions & Trade-offs

*   **ONNX Runtime for Inference**:
    *   *Decision*: We use ONNX Runtime instead of running the model directly in PyTorch/SpeechBrain.
    *   *Trade-off*: This reduces overhead and allows for faster CPU inference and easier deployment, but requires an explicit export step for the model.
*   **Energy-Based VAD**:
    *   *Decision*: A simple custom VAD based on signal energy thresholds.
    *   *Trade-off*: It is extremely fast and has zero external dependencies, but it is less robust to background noise compared to model-based VADs (like Silero).
*   **K-Means Clustering**:
    *   *Decision*: Standard K-Means is used for grouping embeddings.
    *   *Trade-off*: It requires the user to know the number of speakers in advance (`--num-speakers`). It doesn't automatically detect the number of speakers like Spectral Clustering or Agglomerative Hierarchical Clustering would.

## Future Improvements

Given more time, the following improvements would be prioritized:

1.  ** robust VAD**: Replace the energy-based approach with a pre-trained neural VAD (e.g., Silero VAD or Pyannote's segmentation model) to handle noisy environments better.
2.  **Auto-tuning Speaker Count**: Implement Spectral Clustering with eigengap heuristics to automatically determine the number of speakers.
3.  **Re-segmentation**: Add a post-processing step (like Viterbi decoding) to refine the precise start/end boundaries of speaker turns, which are currently coarse due to fixed windowing.
4.  **Overlap Handling**: The current pipeline assumes one speaker at a time. Integrating an overlap-aware model would allow detecting multiple simultaneous speakers.