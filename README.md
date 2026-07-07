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

This is my replication (and partial re-implementation) of *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models* (Araci, 2019). I fine-tuned BERT on the Financial PhraseBank dataset so it can label financial statements as positive, negative, or neutral.

> **Original paper:** [Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.*](https://arxiv.org/abs/1908.10063) — MSc thesis, University of Amsterdam / Naspers Group.

---

## 1. Project Overview

The goal here was to rebuild the core sentiment classification pipeline from the FinBERT paper: fine-tuning BERT with a classification head on top of the `[CLS]` token to predict sentiment on short financial sentences. I want to be upfront that this is a *partial* replication — I replicated the classification fine-tuning step, but not the paper's domain-adaptive pre-training or its ablation studies. Section 7 breaks down exactly what's covered and what isn't.

## 2. Dataset

I used the **Financial PhraseBank** (Malo et al., 2014) — English sentences pulled from financial news articles in the LexisNexis database, annotated by 16 finance and business professionals for sentiment, judged by how the information would affect the mentioned company's stock price.

Specifically, I trained on **`Sentences_AllAgree.txt`**, the subset where all annotators agreed on the label (2,262 of the full 4,845 sentences). Worth flagging early: the paper reports its headline numbers on the *full* dataset, not this subset, so the two accuracy figures aren't directly comparable. More on that in Section 7.

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

I trained this on **Google Colab** (T4 GPU runtime) — see [Section 5](#5-hyperparameters--hardware) and [Section 11](#11-tools--workflow) for more on the setup.

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
- **Classification head:** `BertForSequenceClassification`, a dense layer over the `[CLS]` token output, 3-way softmax (negative / neutral / positive)
- **Tokenizer:** `BertTokenizer` (`bert-base-uncased`), max sequence length **64 tokens**, truncation + padding
- **No further domain pre-training** — I fine-tuned directly on top of the general-purpose `bert-base-uncased` checkpoint rather than pre-training on financial text first (more in Section 7)

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

Here's the final evaluation on the 20% held-out test set (453 examples):

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

**⚠️ A quick caveat before the table:** my 97.35% accuracy should be compared to the paper's **100%-agreement subset** result, not their headline "all data" number — those are two different evaluation sets and shouldn't be read side by side without that context.

| Model | Dataset subset | Accuracy | F1 | Notes |
|---|---|---|---|---|
| Paper — FinBERT (full pipeline) | 100% agreement | 0.97 | 0.95 | Domain-pretrained + discriminative FT + gradual unfreezing |
| Paper — FinBERT (full pipeline) | All data | 0.86 | 0.84 | Headline result of the paper |
| Paper — Vanilla BERT (ablation) | All data | 0.85 | 0.84 | No further pre-training — closest baseline to this project |
| **This project** | **100% agreement** | **0.97** | **0.96 (macro)** | No further pre-training, no train/val split, no discriminative FT/gradual unfreezing |

### What lines up with the paper
- Same base checkpoint (`bert-base-uncased`) and tokenizer
- Same max sequence length (64 tokens)
- Classification via a dense layer on `[CLS]`, 3-way sentiment
- Accuracy on the matching (100%-agreement) subset comes out comparable

### What's missing or done differently here
1. **No domain-adaptive pre-training.** The paper's main contribution is further pre-training BERT on the 46K-document TRC2-financial Reuters corpus before fine-tuning — what they call "FinBERT-domain." I skipped that step, so this project is really closer to the paper's "Vanilla BERT" baseline than to "FinBERT" proper.
2. **No validation split.** The paper uses a 60/20/20 train/validation/test split and picks the best checkpoint across 6 epochs based on validation performance. I used a single 80/20 train/test split with no validation-based checkpoint selection or early stopping.
3. **No catastrophic-forgetting mitigations.** The paper found gradual unfreezing to be the most effective stability technique — accuracy dropped from 0.86 to 0.83 without it — and paired it with slanted triangular learning rates and discriminative fine-tuning (discrimination rate 0.85). I didn't use any of these; the whole model gets fine-tuned uniformly instead.
4. **Fewer epochs.** I trained for 3 epochs versus the paper's 6, and since I didn't hold out a validation set, there's no early-stopping step to compensate.
5. **Full-dataset results not reported here.** I only evaluated on the 100%-agreement subset (2,262 sentences). The paper's main comparison table uses the full 4,845-sentence dataset, which has more label noise and lower agreement — a tougher, more realistic test than the one I ran.
6. **No FiQA regression task.** The paper also evaluates a continuous sentiment-score regression variant on FiQA, measured with MSE and R². This project only tackles the 3-class Financial PhraseBank task.
7. **No baseline comparisons.** The paper benchmarks FinBERT against LSTM+GloVe, LSTM+ELMo, and ULMFit; none of those are included here.
8. **No layer-selection or partial fine-tuning ablations.** The paper's RQ5 and RQ6 experiments, which look at which encoder layer to classify from and how many layers actually need fine-tuning, aren't replicated in this project.

### Why the results still look this good despite those gaps
The paper itself notes that on the 100%-agreement subset, even their Vanilla BERT baseline hits 0.96 accuracy — there's just not much room left for domain pre-training to add value on a cleaner, less ambiguous subset like this one. My 97.35% result is consistent with that.

## 8. What I Learned

Working through this taught me a lot about how much the choice of dataset subset can shape reported results — the same underlying model can look dramatically stronger or weaker depending on which slice of the data you evaluate on, which is exactly what's going on between my 97% number and the paper's full-dataset numbers. I also got a lot more comfortable with the Hugging Face `transformers` API: loading a pretrained checkpoint, wiring up a tokenizer, and attaching a classification head on top of `[CLS]` all clicked in a way that felt abstract before I actually built it. Probably the biggest conceptual takeaway was seeing the real difference between "a model built on BERT" and "a model actually adapted to a domain" — domain pre-training isn't just a nice-to-have, it's doing real work, and skipping it has a measurable cost. And going through the paper's methodology in this much detail gave me a much better feel for the kinds of decisions (validation splits, discriminative learning rates, gradual unfreezing) that separate a quick fine-tuning script from a carefully tuned research result.

## 9. Future Improvements

Things I'm logging for future iterations of this project:

- [x] **Evaluate on the full dataset** (`Sentences_50Agree.txt` / full Financial PhraseBank) — got 82.16% accuracy / 0.80 macro F1, below the paper's own full-dataset numbers, likely due to the missing catastrophic-forgetting mitigations and domain pre-training discussed in Section 7.
- [x] **Visualize feature importance / attention** — my model reproduces the paper's own numeric-reasoning failure case, while correctly handling two examples the paper's model got wrong.
- [ ] **Domain-adaptive pre-training**: further pre-train `bert-base-uncased` (masked LM) on a financial text corpus before fine-tuning, to build a true "FinBERT-domain" equivalent
- [ ] **Try another dataset** — e.g. FiQA Task 1 (regression), or a more recent financial sentiment/news dataset, to test generalization
- [ ] **Compare another model** — e.g. `distilbert-base-uncased`, `roberta-base`, or the official `ProsusAI/finbert` checkpoint, against this fine-tuned baseline
- [ ] **Tune hyperparameters** — learning rate, batch size, epochs, warm-up proportion — with a proper validation split for model selection
- [ ] **Ablation study** — reproduce the paper's Table 4/5 style comparisons (no pre-training vs. task pre-training vs. domain pre-training; with/without slanted triangular LR, discriminative fine-tuning, gradual unfreezing)
- [ ] **Analyze training speed** — time-per-epoch and total training time across different hardware or fine-tuning strategies (e.g. fine-tuning only the last k layers, per the paper's RQ6)
- [ ] **Error analysis** — build a confusion matrix and inspect misclassified examples, similar to the paper's Figure 4 and Section 6.5 examples (`train_full_dataset.py` already saves a confusion matrix CSV as part of the full-dataset run)

## 10. Acknowledgements

Original paper by Dogu Tan Araci (University of Amsterdam / Naspers Group). Dataset: Financial PhraseBank by Malo et al. (2014).

## 11. Tools & Workflow

I built this project using:

- **[Google Colab](https://colab.research.google.com/)** — my training environment (T4 GPU runtime)
- **[Claude Code](https://claude.com/product/claude-code)** — helped me write, structure, and debug the data pipeline, tokenizer, dataset, and model modules
- **[Google Gemini](https://gemini.google.com/)** — helped with research, explanations, and troubleshooting along the way

All final code, experimental decisions, and results in this repository were reviewed and run by me.
