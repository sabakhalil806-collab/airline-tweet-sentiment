# ✈️ Airline Tweet Sentiment Classifier

> A three-class sentiment classifier (negative / neutral / positive) for airline tweets, comparing a classic TF-IDF + Logistic Regression pipeline with a fine-tuned RoBERTa transformer.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-classic%20ML-orange)
![Transformers](https://img.shields.io/badge/Hugging%20Face-RoBERTa-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

**Result at a glance:** the fine-tuned RoBERTa model reaches **macro F1 0.829** (accuracy 0.868) on the held-out test set, compared with **0.754** (accuracy 0.805) for the best classic model.

Task: Intelligent Multi-Class Natural Language Text Sentiment Classifier.

---

## 📌 Overview

This project classifies airline tweets as **negative**, **neutral** or **positive** using the
[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) dataset.

It compares a classic machine-learning pipeline with a fine-tuned tweet-pretrained transformer
(`cardiffnlp/twitter-roberta-base-sentiment-latest`) using a careful evaluation setup: cross-validated decisions, a test set used only once, bootstrap confidence intervals, label-quality checks and a structured error analysis.

---

## ✨ Features

* 🧹 Near-duplicate removal **before** the train/test split (no leakage)
* 🧪 Preprocessing ablation: 7 variants compared with cross-validation
* 🔤 Word and character n-gram TF-IDF, airline and style features
* 📊 Repeated cross-validation for 7 model configurations (mean ± std)
* ⚖️ Neutral-class threshold tuning on out-of-fold predictions
* 🏷️ Label-quality analysis using annotator confidence
* 🤖 Fine-tuned RoBERTa transformer with a class-weighted loss
* 📈 Bootstrap 95 % confidence intervals and paired model comparison
* 🔍 Structured error analysis with automatic error tags
* 🖥️ Interactive Gradio demo

---

## 🖥️ Demo (Gradio)

The notebook ends with an interactive app: type a tweet, choose a model, and see the predicted class with the probability of each class. Below are three tests I ran on the finished app.

### Test 1: Generalization to a new domain (fine-tuned transformer)

A sentence that is **not about airlines** and has no obvious sentiment keywords; the negativity comes only from the meaning.

> *"The train was ninety minutes late and the conductor just shrugged when we asked what was going on."*

<img src="docs/demo-1-transformer-negative.png" alt="Fine-tuned transformer predicts Negative with 99% confidence for a train-delay sentence" width="800">

**Result:** Negative, 99 % confidence. The model transfers beyond airline vocabulary.

### Test 2: Neutral question (classic ML)

A factual question with no emotion. Neutral is the hardest class in this dataset.

> *"@AmericanAir What time does the check-in counter open for flight 221 tomorrow?"*

<img src="docs/demo-2-classic-neutral.png" alt="Classic ML predicts Neutral with 83% confidence for a check-in question" width="800">

**Result:** Neutral, 83 % confidence (negative 15 %, positive 2 %).

### Test 3: Mixed sentiment (fine-tuned transformer)

A polite opening followed by the real complaint.

> *"@SouthwestAir The staff were really polite, but my bag was lost and nobody could help."*

<img src="docs/demo-3-transformer-mixed.png" alt="Fine-tuned transformer predicts Negative with 99% confidence for a mixed-sentiment tweet" width="800">

**Result:** Negative, 99 % confidence. The model weighs the complaint after "but" over the positive start.

### Summary of the three tests

| # | Tweet (short) | Model | Expected | Predicted | Confidence |
|:-:|---|---|:-:|:-:|:-:|
| 1 | Train 90 minutes late, conductor shrugged | Fine-tuned RoBERTa | Negative | ✅ Negative | 99 % |
| 2 | What time does check-in open for flight 221? | Classic ML | Neutral | ✅ Neutral | 83 % |
| 3 | Polite staff, but bag lost and no help | Fine-tuned RoBERTa | Negative | ✅ Negative | 99 % |

> Three hand-picked examples are a sanity check, not an evaluation. The real evidence is the test-set results below.

---

## 📊 Results (held-out test set, 2,837 tweets)

| Model | Accuracy | Macro F1 | 95 % CI (bootstrap) |
|---|:---:|:---:|:---:|
| Majority-class baseline | 0.639 | 0.260 | 0.255 to 0.264 |
| Classic ML (Logistic Regression, word + char n-grams) | 0.805 | 0.754 | 0.735 to 0.772 |
| Classic ML + tuned class weights | 0.815 | 0.759 | 0.740 to 0.777 |
| **Fine-tuned RoBERTa** | **0.868** | **0.829** | **0.811 to 0.844** |

Per-class F1 on the test set:

| Model | Negative | Neutral | Positive |
|---|:---:|:---:|:---:|
| Classic ML | 0.878 | 0.653 | 0.730 |
| Fine-tuned RoBERTa | 0.923 | 0.729 | 0.834 |

The transformer improves macro F1 by about **+0.075** over the best classic model (paired bootstrap 95 % CI +0.059 to +0.092, significant) and improves all three classes.

---

## 🔍 Key Findings

1. **Preprocessing matters less than expected.** Cleaning plus sentiment signals (`!`, `?`, emoticons) scored best (CV macro F1 0.748). Removing stop-words *lowered* the score, and the baseline pipeline with standard stop-words and lemmatization scored 0.716.
2. **Character n-grams help.** Logistic Regression improved from 0.749 (word n-grams) to 0.762 (word + character n-grams). Airline and style features added nothing, and Complement Naive Bayes was clearly worse (0.688).
3. **Tuning and threshold adjustment gave no reliable gain.** GridSearchCV scored below the untuned model, and class-weight tuning moved macro F1 by only +0.005 (confidence intervals overlap).
4. **Label noise limits the score.** On the 70 % of test tweets with annotator confidence ≥ 0.8, the classic model's macro F1 is **0.847** (vs 0.754 on all tweets). Half of the neutral tweets have low annotator confidence.
5. **Errors.** The transformer misclassifies 13.2 % of test tweets. The most common error tags are low label confidence (69 % of errors), negation, mixed sentiment and very short tweets. The largest confusions are negative ↔ neutral.

---

## 🔄 Workflow

```text
Twitter US Airline Sentiment (14,640 tweets)
            │
            ▼
Remove near-duplicates  →  14,181 tweets
            │
            ▼
Stratified 80/20 split  (test set locked away)
            │
            ▼
Preprocessing ablation  →  best variant chosen by CV
            │
            ▼
Feature building  →  word + char n-grams, airline, style features
            │
            ▼
Model comparison (repeated CV)  →  Logistic Regression, SVM, Naive Bayes
            │
            ├──► Tuning + neutral-class thresholds
            │
            ├──► Label-quality analysis
            │
            └──► Fine-tuned RoBERTa transformer
                          │
                          ▼
        Final test evaluation (used once, bootstrap CIs)
                          │
                          ▼
        Error analysis  →  Gradio demo
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Main programming language |
| **scikit-learn** | TF-IDF, Logistic Regression, SVM, cross-validation, tuning |
| **NLTK** | Stop-words and lemmatization |
| **Hugging Face Transformers + PyTorch** | Fine-tuning RoBERTa |
| **pandas / NumPy** | Data handling |
| **matplotlib / seaborn** | Charts |
| **Gradio** | Interactive demo |
| **Kaggle** | Notebook environment and free GPU |

---

## 📁 Project Structure

```text
airline-tweet-sentiment/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── notebooks/
│   └── sentiment-analyzer.ipynb
│
└── docs/
    ├── demo-1-transformer-negative.png
    ├── demo-2-classic-neutral.png
    └── demo-3-transformer-mixed.png
```

---

## ⚙️ How to Run

### Option A: Kaggle (recommended, free GPU)

1. Upload `Notebook/sentiment-analyser.ipynb` to Kaggle and add the *Twitter US Airline Sentiment* dataset.
2. Set **Internet → On**. For the transformer section also set **Accelerator → GPU T4**.
3. Run all cells from top to bottom. Without a GPU the transformer section is skipped and everything else still runs.
4. The last code cell launches the Gradio demo; open the `gradio.live` link it prints.

### Option B: Locally

```bash
git clone https://github.com/sabakhalil806-collab/airline-tweet-sentiment.git
cd airline-tweet-sentiment
pip install -r requirements.txt
# download Tweets.csv from the Kaggle dataset and place it next to the notebook
jupyter notebook notebooks/sentiment-analyzer.ipynb
```

Transformer fine-tuning needs a CUDA GPU. The classic-ML parts run on CPU, but the repeated cross-validation takes a while.

---

## 📚 Dataset

[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) (14,640 tweets, 2015; 64 % negative, 21 % neutral, 15 % positive).

The data is **not included** in this repository. Download it from Kaggle and follow its license terms.

---

## ⚠️ Limitations

* One dataset (US airlines, 2015); results may not transfer to other domains or to current Twitter/X language.
* Labels are noisy, especially for neutral and mixed tweets, which caps the achievable score.
* Auto-tags in the error analysis are heuristics; only the manually tagged sample is reliable.
* Results come from a single split and a single fine-tuning run (one random seed).

---

## 🚀 Future Improvements

* 🤝 Ensemble the classic model and the transformer by averaging their probabilities
* 🎲 Fine-tune with several seeds, and try a larger model or domain-adaptive pretraining
* ⚖️ Use annotator confidence as sample weights instead of filtering labels
* 🧠 Improve handling of negation and sarcasm
* 🧩 Multi-task learning with `negativereason`, or aspect-based sentiment
* 🌐 Deploy the demo permanently (for example on Hugging Face Spaces)

---

## 📄 License

Released under the [MIT License](LICENSE).

---

## 👤 Author

**Saba Khalil**, MS AI 25, NUTECH

⭐ If you find this project useful, consider giving the repository a **star** on GitHub.
