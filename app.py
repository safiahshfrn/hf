import os
import re
import sys

# Core dependency: streamlit (required)
try:
    import streamlit as st
except Exception as e:
    print("Error: streamlit is required to run this app. Install with: pip install streamlit")
    raise

# Optional / commonly used libraries — load gracefully and record missing ones
_missing_pkgs = []

try:
    import pandas as pd
except Exception:
    pd = None
    _missing_pkgs.append('pandas')

try:
    import numpy as np
except Exception:
    np = None
    _missing_pkgs.append('numpy')

try:
    import joblib
except Exception:
    joblib = None
    _missing_pkgs.append('joblib')

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
    _missing_pkgs.append('matplotlib')

try:
    import seaborn as sns
except Exception:
    sns = None
    _missing_pkgs.append('seaborn')

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None
    _missing_pkgs.append('wordcloud')

# NLTK (used for preprocessing). Attempt to import and download required corpora.
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception:
    nltk = None
    stopwords = None
    WordNetLemmatizer = None
    _missing_pkgs.append('nltk')

# If critical optional packages are missing, show a sidebar warning in Streamlit
if _missing_pkgs:
    try:
        st.sidebar.warning('Missing optional packages: ' + ', '.join(_missing_pkgs) +
                           '. Some features may be limited. Install them with pip.')
    except Exception:
        # If Streamlit sidebar is unavailable for some reason, print to stdout
        print('Missing optional packages:', ', '.join(_missing_pkgs))

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Analyzer",
    page_icon="😊",
    layout="wide"
)

# ── Label mapping ─────────────────────────────────────────────
id2label  = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
label2id  = {v: k for k, v in id2label.items()}
EMOTIONS  = list(id2label.values())
EMOJI_MAP = {
    'joy':      '😄',
    'sadness':  '😢',
    'anger':    '😡',
    'fear':     '😨',
    'love':     '❤️',
    'surprise': '😲',
}
COLOR_MAP = {
    'joy':      '#FFD700',
    'sadness':  '#4169E1',
    'anger':    '#DC143C',
    'fear':     '#800080',
    'love':     '#FF69B4',
    'surprise': '#FF8C00',
}

# ── Preprocessing ─────────────────────────────────────────────
@st.cache_resource
def load_nlp_tools():
    # Return safe defaults if NLTK is not available
    if stopwords is None or WordNetLemmatizer is None:
        class _DummyLem:
            def lemmatize(self, w):
                return w
        return set(), _DummyLem()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

def preprocess(text, stop_words, lemmatizer):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words
             if w not in stop_words and len(w) > 1]
    return ' '.join(words)

# ── Model loading ─────────────────────────────────────────────
@st.cache_resource
def load_classical_model():
    # If joblib is not installed, skip classical model loading
    if joblib is None:
        return None, None
    try:
        tfidf = joblib.load('models/tfidf_vectorizer.pkl')
        svm   = joblib.load('models/best_classical_model.pkl')
        return tfidf, svm
    except Exception:
        return None, None

@st.cache_resource
def load_w2v_model():
    try:
        from gensim.models import Word2Vec
        import numpy as np_inner
        w2v = Word2Vec.load('models/word2vec.model')
        lr  = joblib.load('models/lr_word2vec_model.pkl')
        return w2v, lr
    except Exception:
        return None, None

def text_to_w2v_vector(text, w2v_model):
    import numpy as np_inner
    words = text.split()
    vecs = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
    if not vecs:
        return np_inner.zeros(w2v_model.vector_size)
    return np_inner.mean(vecs, axis=0)

@st.cache_resource
def load_bert_model():
    try:
        from transformers import pipeline
        pipe = pipeline(
            'text-classification',
            model='models/best_transformer_model',
            tokenizer='models/best_transformer_model',
            top_k=None
        )
        return pipe
    except Exception as e:
        st.sidebar.error(f"DistilBERT failed to load: {e}")
        return None

@st.cache_data
def load_dataset():
    if pd is None:
        return None
    try:
        return pd.read_csv('data/emotion_dataset_processed.csv')
    except Exception:
        return None

# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.title("😊 Emotion Analyzer")
st.sidebar.markdown("**SAIA 2163 — NLP Final Project**")
st.sidebar.markdown("Team: Syiqin · Ain · Saf")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Text Analyzer", "📊 Data Explorer",
     "📈 Visualizations", "🤖 Model Info"]
)

# ══════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("😊 Social Media Emotion Analyzer")
    st.subheader("SAIA 2163 — NLP Final Project")

    st.markdown("""
    ### What this app does
    This application detects **emotions** in social media text using Natural Language Processing.
    It classifies text into one of **6 emotions**:
    """)

    cols = st.columns(6)
    for i, (emo, emoji) in enumerate(EMOJI_MAP.items()):
        with cols[i]:
            st.metric(label=emoji, value=emo.capitalize())

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### How to use
        1. Go to **Text Analyzer**
        2. Type or paste any social media text
        3. Click **Analyze**
        4. See the predicted emotion and confidence scores
        """)
    with col2:
        st.markdown("""
        ### Dataset
        - **Source:** DAIR-AI Emotion (Twitter)
        - **Size:** 20,000 samples
        - **Classes:** 6 emotions
        - **Best model accuracy:** 93.7% (DistilBERT)
        """)

    st.markdown("---")
    st.markdown("""
    ### Team Members
    | Name | Role |
    |------|------|
    | Syiqin | Dataset & Streamlit App |
    | Ain | NLP Pipeline & Models |
    | Saf | Analysis & Integration |
    """)

# ══════════════════════════════════════════════════════════════
# PAGE 2: TEXT ANALYZER
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Text Analyzer":
    st.title("🔍 Text Emotion Analyzer")
    st.markdown("Enter any social media text to detect its emotion.")

    stop_words, lemmatizer = load_nlp_tools()
    tfidf, svm = load_classical_model()
    bert_pipe  = load_bert_model()
    w2v_model, w2v_lr = load_w2v_model()

    model_choice = st.selectbox(
        "Choose model",
        ["DistilBERT (Best — 93.7%)", "LinearSVC / SVM (90.6%)", "Logistic Regression + Word2Vec (44.7%)"]
    )

    # ── Example sentences ─────────────────────────────────────
    EXAMPLES = {
        "😄 Joy":      "i feel so happy and joyful today laughing with friends enjoying life feeling blessed grateful and full of positive energy",
        "😢 Sadness":  "i feel so lonely and empty inside nobody understands me i cry myself to sleep every night missing people i lost",
        "😡 Anger":    "i am absolutely furious about this situation it is completely unacceptable they betrayed my trust and i feel so angry",
        "❤️ Love":     "i love my partner so much they are my soulmate i feel romantic affection every moment i cherish our relationship deeply",
        "😨 Fear":     "i feel so scared and terrified right now i am shaking with anxiety and dread something awful is going to happen",
        "😲 Surprise": "oh my god i cannot believe this happened i am completely shocked and stunned i never expected this at all wow",
    }

    st.markdown("### 💡 Try an example — click to fill:")
    ex_cols = st.columns(3)
    for i, (label, text) in enumerate(EXAMPLES.items()):
        with ex_cols[i % 3]:
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state['text_input'] = text
                st.rerun()

    user_input = st.text_area(
        "Enter text here:",
        placeholder="e.g. I feel so happy today! This is amazing!",
        height=120,
        key="text_input"
    )

    if st.button("🔍 Analyze", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing..."):

                if "DistilBERT" in model_choice:
                    if bert_pipe is None:
                        st.error("DistilBERT failed to load. Run: pip install torch transformers")
                        st.stop()
                    results = bert_pipe(user_input[:512])[0]
                    scores = {r['label']: r['score'] for r in results}
                    predicted = max(scores, key=scores.get)
                    confidence = scores[predicted]

                elif "Word2Vec" in model_choice:
                    if w2v_model is None or w2v_lr is None:
                        st.error("Word2Vec model failed to load. Run: pip install gensim")
                        st.stop()
                    clean = preprocess(user_input, stop_words, lemmatizer)
                    vec = text_to_w2v_vector(clean, w2v_model).reshape(1, -1)
                    predicted = w2v_lr.predict(vec)[0]
                    proba = w2v_lr.predict_proba(vec)[0]
                    scores = {cls: float(p) for cls, p in zip(w2v_lr.classes_, proba)}
                    confidence = scores[predicted]

                elif "SVM" in model_choice:
                    if tfidf is None or svm is None:
                        st.error("SVM model failed to load. Check models/ folder.")
                        st.stop()
                    clean = preprocess(user_input, stop_words, lemmatizer)
                    vec   = tfidf.transform([clean])
                    predicted  = svm.predict(vec)[0]
                    dec = svm.decision_function(vec)[0]
                    exp_d = np.exp(dec - dec.max())
                    probs = exp_d / exp_d.sum()
                    scores = {cls: float(p) for cls, p in zip(svm.classes_, probs)}
                    confidence = scores[predicted]

                else:
                    st.error("No model loaded. Make sure models/ folder exists.")
                    st.stop()

            # ── Results ──
            emoji = EMOJI_MAP.get(predicted, '❓')

            st.markdown("---")
            st.markdown(f"## Result: {emoji} **{predicted.upper()}**")
            st.progress(confidence)
            st.caption(f"Confidence: {confidence*100:.1f}%")

            # Confidence bar chart
            st.markdown("### Confidence for each emotion")
            if pd is not None and plt is not None and np is not None:
                score_df = pd.DataFrame({
                    'Emotion': list(scores.keys()),
                    'Confidence': list(scores.values())
                }).sort_values('Confidence', ascending=True)

                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.barh(score_df['Emotion'], score_df['Confidence'],
                               color=[COLOR_MAP.get(e, '#888') for e in score_df['Emotion']])
                ax.set_xlim(0, 1)
                ax.set_xlabel('Confidence Score')
                ax.set_title('Emotion Confidence Scores')
                for bar, val in zip(bars, score_df['Confidence']):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{val:.3f}', va='center', fontsize=9)
                st.pyplot(fig)
                plt.close()
            else:
                st.write(scores)

            # Key words that influenced prediction
            clean_for_words = preprocess(user_input, stop_words, lemmatizer)
            if "SVM" in model_choice and tfidf is not None:
                words_in_vocab = [w for w in clean_for_words.split() if w in tfidf.vocabulary_]
                if words_in_vocab:
                    st.markdown("### 🔑 Words that influenced prediction")
                    st.write(", ".join([f"`{w}`" for w in words_in_vocab]))
            elif "Word2Vec" in model_choice and w2v_model is not None:
                words_in_vocab = [w for w in clean_for_words.split() if w in w2v_model.wv]
                if words_in_vocab:
                    st.markdown("### 🔑 Words that influenced prediction")
                    st.write(", ".join([f"`{w}`" for w in words_in_vocab]))

# ══════════════════════════════════════════════════════════════
# PAGE 3: DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")

    df = load_dataset()
    if df is None:
        st.error("Dataset not found. Run the notebook first to generate data/emotion_dataset_processed.csv")
        st.stop()

    st.markdown(f"**Total samples:** {len(df):,} | **Emotions:** {df['emotion'].nunique()}")

    st.markdown("### Sample Data")
    st.dataframe(df[['text', 'emotion', 'clean_text']].sample(10, random_state=42), use_container_width=True)

    st.markdown("### Dataset Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Class distribution**")
        counts = df['emotion'].value_counts().reset_index()
        counts.columns = ['Emotion', 'Count']
        st.dataframe(counts, use_container_width=True)

    with col2:
        st.markdown("**Text length stats (words)**")
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
        st.dataframe(df.groupby('emotion')['word_count']
                     .describe()[['mean','min','max']]
                     .round(1), use_container_width=True)

    st.markdown("### Class Distribution Chart")
    # Class distribution plot (if plotting libs available)
    if plt is not None and sns is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        emotion_counts = df['emotion'].value_counts()
        palette = sns.color_palette('viridis', len(emotion_counts))

        axes[0].bar(emotion_counts.index, emotion_counts.values, color=palette)
        axes[0].set_title('Count per Emotion')
        axes[0].set_xlabel('Emotion')
        axes[0].set_ylabel('Count')

        axes[1].pie(emotion_counts.values, labels=emotion_counts.index,
                    autopct='%1.1f%%', colors=palette)
        axes[1].set_title('Distribution (%)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info('Plotting requires matplotlib and seaborn. Install them to see charts.')

# ══════════════════════════════════════════════════════════════
# PAGE 4: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.title("📈 Visualizations")

    viz_map = {
        "Class Distribution":       "visualizations/class_distribution.png",
        "Word Cloud (All Emotions)": "visualizations/wordcloud_all.png",
        "Word Clouds per Emotion":   "visualizations/wordclouds_per_emotion.png",
        "Top 20 TF-IDF Words":       "visualizations/top20_tfidf_words.png",
        "Model Comparison":          "visualizations/model_comparison.png",
        "Confusion Matrices":        "visualizations/confusion_matrices.png",
        "Text Length Distribution":  "visualizations/text_length_distribution.png",
    }

    for title, path in viz_map.items():
        if os.path.exists(path):
            st.markdown(f"### {title}")
            st.image(path, use_container_width=True)
            st.markdown("---")
        else:
            st.info(f"**{title}** — not generated yet (run the notebook first)")

# ══════════════════════════════════════════════════════════════
# PAGE 5: MODEL INFO
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Model Info":
    st.title("🤖 Model Information")

    st.markdown("### Model Performance Summary")
    results_data = {
        'Model': ['DistilBERT (fine-tuned)', 'LinearSVC / SVM (TF-IDF)',
                  'Logistic Regression (TF-IDF)', 'Naive Bayes (TF-IDF)',
                  'Logistic Regression (Word2Vec)'],
        'Feature': ['Raw text', 'TF-IDF', 'TF-IDF', 'TF-IDF', 'Word2Vec'],
        'Accuracy': [0.9370, 0.9055, 0.9000, 0.7578, 0.4470],
        'Macro-F1': [0.9111, 0.8704, 0.8685, 0.5754, 0.3940],
    }
    if pd is not None:
        results_df = pd.DataFrame(results_data)
        results_df['Accuracy'] = results_df['Accuracy'].map('{:.2%}'.format)
        results_df['Macro-F1'] = results_df['Macro-F1'].map('{:.2%}'.format)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    else:
        # Fallback display when pandas is missing
        st.info('pandas not installed — showing plain results')
        for m, f, a, s in zip(results_data['Model'], results_data['Feature'], results_data['Accuracy'], results_data['Macro-F1']):
            st.markdown(f"**{m}** — Feature: {f} — Accuracy: {a:.2%} — Macro-F1: {s:.2%}")

    st.markdown("---")
    st.markdown("""
    ### Model Details

    **DistilBERT (Best Model)**
    - Architecture: Transformer with 6 attention layers, 66M parameters
    - Pre-trained on BookCorpus + English Wikipedia
    - Fine-tuned for 4 epochs on 16,000 training samples
    - Achieves **93.7% accuracy** — best overall

    **LinearSVC / SVM**
    - Support Vector Machine with linear kernel
    - Features: TF-IDF (10,000 features, unigrams + bigrams)
    - Class weighting: balanced (handles imbalance)
    - Achieves **90.6% accuracy** — best classical model

    **Feature Extraction Methods**
    - **TF-IDF:** Weighs words by how unique they are to a document. Bigrams capture phrases like "feel sad" or "so happy"
    - **Word2Vec:** Neural embeddings that capture semantic similarity. Averaged across words to form a document vector

    ### Why Word2Vec + LR scored low (44.7%)
    Averaging word vectors loses word order and context — "not happy" and "happy" get similar vectors.
    For short social media text, TF-IDF and transformers outperform simple averaged embeddings.
    This contrast is valuable: it shows why model and feature choice matters in NLP.

    ### Training Setup
    - Dataset: DAIR-AI Emotion (20,000 Twitter samples)
    - Train/test split: 80/20 stratified
    - Hardware: Apple Silicon MPS / NVIDIA CUDA
    """)
