Short Report: AfriSenti Multilingual Sentiment Analysis

(Placeholder) This report will be populated after running the notebook experiments. It will include:

- Dataset overview: number of tweets, language breakdown (Swahili, Amharic, English), and label distributions.
- Model architectures: BiLSTM baseline and XLM-R/AfriBERTa fine-tuning details.
- Training setup: hyperparameters, early stopping, gradient clipping, and training curves.
- Evaluation: accuracy, macro F1, per-class F1, ROC-AUC, confusion matrices, and per-language breakdown.
- Ablation study results: effect of batch size, learning rate, and sequence length.
- Cross-lingual transfer: train-on-Swahili, test-on-English/Amharic results and analysis.
- Conclusions and recommendations for production deployment and further work.

I'll populate this file with results and figures once the notebook experiments are executed.

## Demo run results (synthetic data)

Note: the following results come from a synthetic demo run (not the real AfriSenti dataset). They were generated to validate the pipeline and create example artifacts for the presentation. Replace with real results after running the notebook on the AfriSenti CSVs.

- Accuracy: 1.0
- Macro F1: 1.0

Per-class metrics (precision / recall / f1-score / support):

- negative: 1.0 / 1.0 / 1.0 / 30
- neutral : 1.0 / 1.0 / 1.0 / 31
- positive: 1.0 / 1.0 / 1.0 / 29

Artifacts produced by the demo (saved under `eda_outputs/`):

- `confusion_demo.png` — confusion matrix for the demo classifier
- `metrics_demo.json` — full metrics and classification report
- `examples_demo.json` — example inputs, predictions, and probabilities
- Multiple EDA plots: `heatmap_label_by_language.png`, `scatter_token_vs_char.png`, `hist_char_len.png`, `hist_token_len.png`, `pca_tfidf_scatter.png`, `correlation_heatmap.png`, and per-language label histograms.

To generate real results, place AfriSenti CSV(s) in `data/` (columns: `text`, `label`, `language`) and run the notebook `AfriSenti_sentiment_analysis.ipynb` end-to-end, then re-run `tools/generate_additional_plots.py` and `tools/generate_presentation.py` to update artifacts and the PPTX.