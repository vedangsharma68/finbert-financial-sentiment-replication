# FinBERT Replication — Financial Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-Hugging%20Face-yellow)
![Model](https://img.shields.io/badge/Base%20Model-bert--base--uncased-blueviolet)
![Accuracy](https://img.shields.io/badge/Accuracy%20(100%25%20agreement)-97.35%25-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy%20(full%20dataset)-82.16%25-green)
![Colab](https://img.shields.io/badge/Trained%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Replication%20In%20Progress-orange)

A replication and partial re-implementation of *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models* (Araci, 2019), fine-tuning BERT on the Financial PhraseBank dataset to classify financial statements as positive, negative, or neutral.

> **Original paper:** [Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.*](https://arxiv.org/abs/1908.10063) — MSc thesis, University of Amsterdam / Naspers Group.

---

## 1. Project Overview

This project re-implements the core sentiment classification pipeline of the FinBERT paper: fine-tuning a BERT model with a classification head on top of the [CLS] token for sentiment prediction on short financial sentences. It is a partial replication: it replicates the classification fine-tuning step but not the domain-adaptive pre-training of the paper, or its ablation studies (see Section 7 for the full breakdown).

## 2. Dataset

**Financial PhraseBank** (Malo et al., 2014) – English sentences from financial news articles in the LexisNexis database, annotated by 16 finance/business professionals for sentiment (negative / neutral / positive) with respect to the impact of the information on the mentioned company’s stock price.

**`Sentences_AllAgree.txt`** is used for this project (the subset where **all annotators agreed** on the label (2,262 out of the full 4,845 sentences). This is an important point: the paper reports results on this subset *independently* of the full dataset, and the two are not directly comparable (see Section 7).

| | |
|---|---|
| Source file | `data/Sentences_AllAgree.txt` |
| Format | `sentence@label`, ISO-8859-1 encoded |
| Total examples | 2,262 |
| Class distribution | ~61% neutral, ~25% positive, ~13% negative (per paper Table 1) |
| Split used | 80% train / 20% test, stratified, `random_state=42` |

## 3. Setup Instructions

```bash
# Clone the repo
git clone <your-repo-url>
cd finbert-replicated

# Create environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers pandas scikit-learn

# Place the dataset
# Download Financial PhraseBank and put Sentences_AllAgree.txt in data/

# Run the pipeline
python src/data_pipeline.py        # loads + splits data
python src/tokenizer_pipeline.py   # tokenizes with bert-base-uncased
python src/dataset_factory.py      # wraps into a PyTorch Dataset
python src/model_pipeline.py       # initializes BertForSequenceClassification
# python src/train.py              # [add your training script + instructions here]
```

This project was trained on **Google Colab** (T4 GPU runtime) — see [Section 6](#6-hyperparameters--hardware) and [Section 12](#12-tools--workflow) for details.

**Project structure:**
```
finbert-replicated-model/
├── data/
│   └── Sentences_AllAgree.txt
├── finbert_replicated_model/       # saved model artifacts
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── notes/
│   └── summary.md
├── src/
│   ├── data_pipeline.py
│   ├── dataset_factory.py
│   ├── model_pipeline.py
│   └── tokenizer_pipeline.py
├── paper.pdf
└── README.md
```

## 4. Model & Implementation

- **Base model:** `bert-base-uncased` (12 layers, hidden size 768, 12 attention heads, 110M params) via Hugging Face `transformers`
- **Classification head:** `BertForSequenceClassification` with a dense layer over the `[CLS]` token output, 3-way softmax (negative / neutral / positive)
- **Tokenizer:** `BertTokenizer` (`bert-base-uncased`), max sequence length **64 tokens**, truncation + padding
- **No further domain pre-training was performed** — the model starts from the general-purpose `bert-base-uncased` checkpoint and is fine-tuned directly on Financial PhraseBank (see Section 7)

## 5. Hyperparameters & Hardware

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Max sequence length | 64 tokens |
| Train/test split | 80% / 20%, stratified |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Adam epsilon | 1e-8 |
| Gradient clipping | max_norm = 1.0 |
| Epochs | 3 |
| Discriminative fine-tuning | No |
| Gradual unfreezing | No |
| Hardware | Google Colab (T4 GPU) |
| Total training time | ~15 minutes |

## 6. Results

Final evaluation on the 20% held-out test set (453 examples):

**Overall accuracy: 97.35%**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Negative (0) | 0.92 | 0.96 | 0.94 | 56 |
| Neutral (1) | 0.99 | 0.99 | 0.99 | 276 |
| Positive (2) | 0.97 | 0.94 | 0.95 | 121 |
| **Accuracy** | | | **0.97** | 453 |
| **Macro avg** | 0.96 | 0.97 | 0.96 | 453 |
| **Weighted avg** | 0.97 | 0.97 | 0.97 | 453 |

## 7. Comparison with the Paper

**⚠️ Important context:** this project's 97.35% accuracy should be compared against the paper's **100%-agreement subset** results, not its headline "all data" number — they're evaluated on different data.

| Model | Dataset subset | Accuracy | F1 | Notes |
|---|---|---|---|---|
| Paper — FinBERT (full pipeline) | 100% agreement | 0.97 | 0.95 | Domain-pretrained + discriminative FT + gradual unfreezing |
| Paper — FinBERT (full pipeline) | All data | 0.86 | 0.84 | Headline result of the paper |
| Paper — Vanilla BERT (ablation) | All data | 0.85 | 0.84 | No further pre-training — closest baseline to this project |
| **This project** | **100% agreement** | **0.97** | **0.96 (macro)** | No further pre-training, no train/val split, no discriminative FT/gradual unfreezing |
| **This project** | **Full dataset (50Agree)** | **0.82** | **0.80 (macro)** | Same hyperparams as above, now with a proper val split + checkpoint selection |

### What matches the paper
- Same base checkpoint (`bert-base-uncased`) and tokenizer
- Same max sequence length (64 tokens)
- Classification via a dense layer on `[CLS]`, 3-way sentiment
- Comparable accuracy on the matching (100%-agreement) subset

### What's missing or implemented differently
1. **No domain-adaptive pre-training.** The main contribution of the paper is the further pre-training of BERT on the 46K-document TRC2-financial Reuters corpus before fine-tuning, which they call "FinBERT-domain." This is not present in this project. This project resembles the paper's "Vanilla BERT" approach more than "FinBERT."
2. **No validation split.** The paper uses a 60/20/20 train/validation/test split and chooses the best checkpoint over 6 epochs on the validation set. This project uses a single 80/20 train/test split and does not use validation-based model selection, so there is no early-stopping or checkpoint selection step. 
3. **No catastrophic-forgetting mitigations.** TThe paper found that gradual unfreezing was the most effective technique for stability; accuracy dropped from 0.86 to 0.83 without it. This technique was applied with slanted triangular learning rates and discriminative fine-tuning (discrimination rate 0.85). This project does not use discriminative fine-tuning or gradual unfreezing. Instead, the whole model is fine-tuned uniformly.
4. **Fewer epochs than the paper.** This project trains for 3 epochs; the paper trains for 6 epochs and selects the best checkpoint on a validation set. Without a validation split, there is no equivalent early-stopping or model-selection step here.
5. **Full-dataset results not reported.** This project evaluates only on the 100%-agreement subset of 2,262 sentences. The paper's main comparison table uses the full dataset of 4,845 sentences, which includes more label noise and lower agreement. This makes for a tougher and more realistic test.
6. **No FiQA regression task.** The paper also assesses a continuous sentiment score regression variant on the FiQA dataset, measuring MSE and R². This project only addresses the 3-class Financial PhraseBank task. 
7. **No baseline comparisons.** The paper compares FinBERT against LSTM+GloVe, LSTM+ELMo, and ULMFit baselines; none of these are included in this project.
8. **No layer-selection or partial fine-tuning ablations.** The paper's RQ5 and RQ6 experiments—which determine which encoder layer to classify from and how many layers require fine-tuning—are not replicated in this project. 

### Why results might still look "as good" despite these gaps
The paper points out that on the 100%-agreement subset, even their Vanilla BERT baseline achieves 0.96 accuracy. There is very little room for further pre-training to improve on this cleaner and less ambiguous subset. This is in line with this project's 97.35% result.

### What the full-dataset run reveals
On the full dataset, however, the results change: this project's 82.16% accuracy and 0.80 macro F1 score fall below both the paper's Vanilla BERT ablation (0.85 / 0.84) and their complete FinBERT pipeline (0.86 / 0.84). This gap was not visible when looking only at the 100%-agreement subset. A few likely contributing factors, in order of expected impact, are:
- **No discriminative fine-tuning / gradual unfreezing / slanted triangular LR.** -  Table 5 of the paper shows that these strategies are worth \~3 points of accuracy on their own. Thus, their omission from this approach is the most probable single cause of the drop in accuracy.
- **No domain pre-training on TRC2-financial.** - While the accuracy of the “Vanilla BERT” models is slightly higher than that of this approach, the two strategies are not equal to one another in performance.
- **3 epochs vs. the paper's 6**, - and less training data (though not substantially less: 60% of 4,845 ≈ 2,900; the paper used \~3,101 examples).
- **Class imbalance interacting with a shorter training run** -  both the negative and positive classes underperform relative to the neutral class in both models, which is expected due to the inherent difficulty of distinguishing between positive and negative sentiments (Section 6.5, Figure 4 of the paper).

This is a good result for the project in that it demonstrates that while the accuracy of 97% was obtained using the filtered dataset, it is not a result of the omissions of strategies used in the original paper.

## 8. What I Learned

*By observing the various uses of pre-trained models discussed in the paper, it is possible to gain an understanding of the performance of those models. Furthermore, it is also possible to gain an understanding of how filtering the datasets according to some criterion can impact the accuracy of the results obtained by researchers using those models. In order to effectively use these models, a person must learn of the functions of the Hugging Face ' transformers API. Furthermore, it is also possible to understand the difference between utilizing a model that incorporates BERT elements versus using a BERT model that has been fine-tuned for a specific field. Finally, developers of these models are able to gain an understanding of the methods used to develop and fine-tune these models.*

## 9. Future Improvements

Extensions logged for future iterations of this project:

- [x] **Evaluate on the full dataset** (`Sentences_50Agree.txt` / full Financial PhraseBank) — 82.16% accuracy / 0.80 macro F1, below the paper's own full-dataset numbers; see [Section 13.1](#131-full-dataset-evaluation) for the likely reasons why.
- [x] **Visualize feature importance / attention** — reproduces the paper's own numeric-reasoning failure case, while correctly handling two examples the paper's model got wrong; see [Section 13.2](#132-feature-importance--attention-visualization).
- [ ] **Domain-adaptive pre-training**: further pre-train `bert-base-uncased` (masked LM) on a financial text corpus before fine-tuning, to build a true "FinBERT-domain" equivalent
- [ ] **Try another dataset** — e.g. FiQA Task 1 (regression), or a more recent financial sentiment/news dataset, to test generalization
- [ ] **Compare another model** — e.g. `distilbert-base-uncased`, `roberta-base`, or the official `ProsusAI/finbert` checkpoint, against this fine-tuned baseline
- [ ] **Tune hyperparameters** — learning rate, batch size, epochs, warm-up proportion — with a proper validation split for model selection
- [ ] **Ablation study** — reproduce the paper's Table 4/5 style comparisons (no pre-training vs. task pre-training vs. domain pre-training; with/without slanted triangular LR, discriminative fine-tuning, gradual unfreezing)
- [ ] **Analyze training speed** — time-per-epoch and total training time across different hardware or fine-tuning strategies (e.g. fine-tuning only the last k layers, per the paper's RQ6)
- [ ] **Error analysis** — build a confusion matrix and inspect misclassified examples, similar to the paper's Figure 4 and Section 6.5 examples (`train_full_dataset.py` already saves a confusion matrix CSV as part of the full-dataset run)

## 10. Limitations

**Dataset scope**
- Only the 100%-agreement subset (2,262 sentences) was evaluated in the original run — the "easiest" slice of Financial PhraseBank, since it excludes every sentence annotators disagreed on. This inflated the headline accuracy relative to the paper's "all data" number: the full-dataset run (Section 13.1) confirms this directly, scoring 82.16% accuracy / 0.80 macro F1 — noticeably below both the AllAgree subset (97.35%) *and* the paper's own full-dataset numbers (0.85–0.86 accuracy), likely because this project's training still lacks the paper's catastrophic-forgetting mitigations (see below).
- Neither run touches the FiQA dataset, so there's no evidence on how the model performs on a continuous-sentiment regression task, or on tweets/headlines rather than full sentences.

**Model & training methodology**
- No domain-specific pre-training corpus (TRC2-financial) was used, so this is architecturally a fine-tuned general-purpose BERT, not a true "FinBERT" in the sense the paper defines it. The paper's central claim — that domain-adaptive pre-training helps — is untested here.
- No catastrophic-forgetting mitigations (discriminative fine-tuning, gradual unfreezing, slanted triangular learning rates) were applied in the original AllAgree run. The paper's own ablation (Table 5) shows these techniques matter — dropping them cost 3 points of accuracy in their experiments — so this is a real, not cosmetic, methodological gap.
- No validation set was carved out in the original run, so there was no checkpoint selection; whatever the state was after 3 fixed epochs is what got evaluated. `train_full_dataset.py` fixes this with a proper 60/20/20 split and best-checkpoint restoration, but that hasn't been run yet either.
- A single train/test split (`random_state=42`) was used rather than the paper's 10-fold cross-validation for some experiments, so reported metrics carry more split-to-split variance than the paper's numbers.
- Trained for 3 epochs vs. the paper's 6 — it's unclear without a validation curve whether 3 epochs under- or over-fits relative to the paper's setup.

**Interpretability**
- Feature-importance analysis (Section 13.2) covers only 3 hand-picked examples carried over from the paper — useful for direct comparison, but far too small a sample to characterize the model's failure modes systematically. A proper error analysis would need attention/saliency plots across a random sample of the actual misclassifications from the full-dataset confusion matrix, not just 3 curated sentences.
- Attention weights and gradient-based saliency are both known to be imperfect proxies for "true" model reasoning (a long-standing critique in interpretability research) — they're useful for spot-checking failure patterns, not as a rigorous explanation of model behavior.

**Generalization**
- All evaluation is on the same source (LexisNexis financial news, pre-2014) the paper used. Nothing here tests whether the model generalizes to more recent financial text, social media, or non-English content.
- No comparison against alternative architectures (DistilBERT, RoBERTa, the official `ProsusAI/finbert` checkpoint) exists yet, so it's unknown whether the gains here come from BERT specifically or would hold with a smaller/different backbone.

## 11. Acknowledgements

Original paper by Dogu Tan Araci (University of Amsterdam / Naspers Group). Dataset: Financial PhraseBank by Malo et al. (2014).

## 12. Tools & Workflow

This project was built using the following tools:

- **[Google Colab](https://colab.research.google.com/)** — training environment (T4 GPU runtime) used to fine-tune the model
- **[Claude Code](https://claude.com/product/claude-code)** — used to help write, structure, and debug the data pipeline, tokenizer, dataset, and model modules
- **[Google Gemini](https://gemini.google.com/)** — used to assist with research, explanation, and general troubleshooting during development

All final code, experimental decisions, and results in this repository were reviewed and run by the author.

## 13. Extended Experiments (In Progress)

Two of the extensions from Section 9 have code ready, pending a Colab run:

### 13.1 Full-dataset evaluation
- **Scripts:** `src/full_dataset_pipeline.py`, `src/train_full_dataset.py`
- **What it does:** Trains on the complete Financial PhraseBank (`Sentences_50Agree.txt`, all 4,845 sentences) using the *same* hyperparameters as the AllAgree run (batch 32, AdamW, lr 2e-5, eps 1e-8, grad clip 1.0, 3 epochs) so "which dataset" is the only variable that changes. Adds a proper 60/20/20 train/val/test split with validation-based checkpoint selection, matching the paper's methodology more closely than the original run.
- **To run:**
  ```bash
  # download Sentences_50Agree.txt into data/, then:
  python src/train_full_dataset.py
  ```
- **Expected outcome:** accuracy should land somewhere between the paper's Vanilla BERT ablation (0.85) and FinBERT-domain (0.86) on the full dataset — likely lower than this project's 97.35% AllAgree result, since the full dataset includes the harder, lower-agreement sentences.
- **Results (Colab T4, 3 epochs, same hyperparameters as the AllAgree run):**

  | Epoch | Train loss | Val loss | Val acc | Val macro F1 | Time |
  |---|---|---|---|---|---|
  | 1/3 | 0.7427 | 0.5287 | 0.7964 | 0.7588 | 38.3s |
  | 2/3 | 0.4859 | 0.3936 | 0.8338 | ~0.81 | 38.1s |
  | 3/3 (best checkpoint) | 0.2633 | 0.3929 | 0.8389 | 0.8206 | 37.4s |

  **Final test set (best val checkpoint, 970 held-out examples):**

  | Metric | Value |
  |---|---|
  | Test loss | 0.4342 |
  | Test accuracy | **0.8216** |
  | Test macro F1 | **0.8020** |

  | Class | Precision | Recall | F1-score | Support |
  |---|---|---|---|---|
  | Negative | 0.70 | 0.88 | 0.78 | 121 |
  | Neutral | 0.90 | 0.82 | 0.86 | 576 |
  | Positive | 0.74 | 0.79 | 0.77 | 273 |
  | **Accuracy** | | | **0.82** | 970 |
  | **Macro avg** | 0.78 | 0.83 | 0.80 | 970 |
  | **Weighted avg** | 0.83 | 0.82 | 0.82 | 970 |

  Total training time (3 epochs): **~114 seconds** on Colab T4. See [Section 7](#7-comparison-with-the-paper) for how this compares to the paper's own full-dataset numbers, and what likely explains the gap.

### 13.2 Feature importance / attention visualization
- **Script:** `src/feature_importance.py`
- **What it does:** For each of the paper's own Section 6.5 failure examples (the euro pre-tax loss sentence, the Fixed-to-Mobile sentence, the coated paper sentence), generates two plots per example: (1) how much the `[CLS]` token attends to each input token in the final encoder layer, and (2) gradient×input saliency for the predicted class. No extra dependencies beyond `transformers`, `torch`, and `matplotlib` — no `captum` needed.
- **To run:**
  ```bash
  python src/feature_importance.py --model_dir finbert_replicated_model
  ```
- **What to look for:** the paper's FinBERT fails Example 1 because it doesn't do the arithmetic comparison between "0.3 million" and "2.2 million." Worth checking whether your model's attention actually lands on those numeric tokens, or skips past them the same way.
- **Results (run against `finbert_replicated_model`, the AllAgree-trained model):**

  | Example | True label | Predicted | Notes |
  |---|---|---|---|
  | 1 — "Pre-tax loss totaled euro 0.3 million, compared to a loss of euro 2.2 million..." | Positive | **Negative** (probs: neg 0.863, neu 0.007, pos 0.131) | **Same failure mode as the paper.** The model doesn't correctly weigh the *improvement* (smaller loss) and instead reacts to "loss" appearing twice — exactly the numeric-comparison failure the paper describes in Section 6.5. |
  | 2 — "This implementation is very important to the operator, since it is about to launch its Fixed to Mobile convergence service in Brazil" | Neutral | **Neutral (correct)** (probs: neg 0.006, neu 0.967, pos 0.027) | The paper's own FinBERT got this one wrong (predicted positive); this run gets it right. |
  | 3 — "The situation of coated magazine printing paper will continue to be weak." | Negative | **Negative (correct)** (probs: neg 0.915, neu 0.009, pos 0.076) | The paper's own FinBERT got this one wrong too (predicted neutral); this run gets it right. |

  Plots saved to `feature_importance_plots/` (attention + saliency PNG per example).

  **Takeaway:** this model reproduces the paper's exact numeric-reasoning blind spot (Example 1) — a good sign the failure mode is a genuine property of fine-tuned BERT on this task, not an artifact of one particular training run — while doing *better* than the paper's model on the two examples that hinge on distinguishing an objective statement from an actual sentiment signal (Examples 2 and 3). That's a reasonable result given this run also skipped domain pre-training and the paper's catastrophic-forgetting mitigations: strong on qualitative tone, weak on quantitative comparison, which BERT's tokenizer and attention mechanism aren't well suited to in the first place (numbers like "0.3" and "2.2" are subword-tokenized and have no inherent magnitude relationship in embedding space).

Both extensions are now complete — see [Section 7](#7-comparison-with-the-paper) for the full-dataset comparison against the paper's numbers, and the takeaway above for what the failure-case analysis shows about where this model's reasoning breaks down.
