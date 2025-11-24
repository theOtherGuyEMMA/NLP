# Short Report: Swahili Speech Classification

## Dataset Overview
- Source: Mozilla Common Voice v11 (Swahili subset) or Swahili Words Speech-Text Parallel Dataset.
- Audio: Variable durations, typically sampled at 16 kHz.
- Labels: For Common Voice, derive classes from frequent words in transcripts (proxy keyword spotting) or custom intent mapping. Swahili Words datasets may have explicit class folders.
- Metadata inspection includes sample rate, duration distribution, and label balance. Figures saved under `figures/` (e.g., `metadata_overview.png`).

## Preprocessing
- Resample audio to `16 kHz` for consistency.
- Feature extraction: MFCC (e.g., 40-coeff) and Mel-spectrograms (64–128 mel bins).
- Normalization: zero-mean, unit-variance per feature map.
- Augmentation: additive Gaussian noise, pitch shifting (+2 semitones), and optional time-stretching for robustness.

## Model Architecture
### 1D CNN Baseline
- Input: feature maps `[channels × time]` from MFCC/Mel.
- Layers: stacked Conv1D + ReLU + MaxPool; Dropout; AdaptiveAvgPool; Linear classifier.
- Regularization: Dropout (e.g., 0.3), ReduceLROnPlateau scheduler, Early Stopping.

### Wav2Vec2 Fine-Tuning (Classification)
- Pretrained `facebook/wav2vec2-base` with a classification head.
- Tokenization via `Wav2Vec2Processor` (raw waveform input); fine-tune for 10–15 epochs on GPU.
- Optional transfer learning: warm-up on English Common Voice, then fine-tune on Swahili.

## Training Protocol
- Epochs: 10–15 (shorter for exploration, longer for final runs).
- Monitoring: training/validation loss and accuracy per epoch.
- Early stopping patience: 3–5 epochs without improvement.
- Optimizer: Adam (`1e-3` CNN; `2e-5` W2V2).

## Evaluation Results
- Metrics: Accuracy, Macro-F1, Confusion Matrix, Classification Report.
- Embeddings visualization: PCA and t-SNE on penultimate layer features to inspect class separability.
- Example figures: `cnn1d_confusion.png`, `embeddings_pca_tsne.png`.

## Exploratory Experiments
- Sampling rate comparison: `8 kHz` vs `16 kHz` shows trade-offs between performance and compute.
- Spectrogram resolution: varying mel bins (40, 64, 128) affects detail vs overfitting; mid-range (64–128) often best.
- Transfer from English: limited direct benefit without label alignment, but improves feature robustness.

## Robustness and Generalization
- Augmentations increase resilience to noise and pitch variations; consider room impulse response simulation for reverberation robustness.
- Cross-speaker generalization: stratified splits and speaker-wise validation recommended.
- Domain shifts: re-evaluate on different recording devices and environments; employ test-time augmentation.

## Extension: Speech-to-Text + Sentiment Pipeline
- ASR: Whisper or Wav2Vec2 for transcription.
- Translation (Swahili→English): `Helsinki-NLP/opus-mt-sw-en` for downstream English sentiment.
- Sentiment: `cardiffnlp/twitter-roberta-base-sentiment` or domain-specific models.
- Pipeline: audio → ASR → (optional translation) → sentiment classifier.

## Key Insights
- CNN on MFCC/Mel provides a strong baseline with low compute.
- Wav2Vec2 fine-tuning benefits from large-scale pretraining and can outperform CNN given enough data and GPU.
- Class definition and label quality critically impact results; clear keyword/intent mapping is essential for Common Voice.
- Robustness requires targeted augmentation and evaluation under varied conditions.

## Next Steps
- Solidify label schema (keyword list or intent classes) for Swahili.
- Scale training and perform hyperparameter search (kernel sizes, layers, activations).
- Add speaker-wise splits; collect additional noisy/real-world samples.
- Deploy pipeline endpoints for classification and ASR+sentiment in production.