# Social Media Emotion Analyzer
### SAIA 2163 — NLP Final Project
**Team:** Syiqin · Ain · Saf

---

## Overview
An NLP application that detects emotions in social media text using machine learning.
Classifies text into 6 emotions: **joy, sadness, anger, fear, love, surprise**

**Dataset:** [DAIR-AI Emotion](https://huggingface.co/datasets/dair-ai/emotion) — 20,000 Twitter samples

---

## Models Trained
| Model | Feature | Accuracy |
|-------|---------|----------|
| Naive Bayes | TF-IDF | ~78% |
| Logistic Regression | TF-IDF | ~85% |
| LinearSVC (SVM) | TF-IDF | ~87% |
| Logistic Regression | Word2Vec | ~75% |
| **DistilBERT** | Raw text | **~92%** |

---

## Setup

### Requirements
- Python 3.9+
- Works on: Windows (CUDA), macOS (MPS/CPU), Linux (CUDA/CPU)

### Install dependencies
```bash
pip install -r requirements.txt
```

### For NVIDIA GPU (CUDA) — faster training
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Run the Streamlit App
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

---

## Run the Notebook
Open `NLP_Emotion_Final.ipynb` in:
- **Jupyter:** `jupyter notebook NLP_Emotion_Final.ipynb`
- **VS Code:** Open file, select Python kernel
- **Google Colab:** Upload file → Runtime → Run all

The notebook auto-detects your GPU (CUDA/MPS/CPU).

---

## Project Structure
```
.
├── NLP_Emotion_Final.ipynb   # Main notebook (NLP pipeline)
├── app.py                    # Streamlit web application
├── nlp_utils.py              # Shared preprocessing utilities
├── requirements.txt
├── README.md
├── data/
│   └── emotion_dataset_processed.csv
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── best_classical_model.pkl
│   ├── lr_word2vec_model.pkl
│   ├── word2vec.model
│   └── best_transformer_model/
└── visualizations/
    ├── class_distribution.png
    ├── wordcloud_all.png
    ├── wordclouds_per_emotion.png
    ├── top20_tfidf_words.png
    ├── model_comparison.png
    └── confusion_matrices.png
```
