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

# NLTK
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

if _missing_pkgs:
    try:
        st.sidebar.warning('Missing optional packages: ' + ', '.join(_missing_pkgs) +
                           '. Some features may be limited. Install them with pip.')
    except Exception:
        print('Missing optional packages:', ', '.join(_missing_pkgs))

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Analyzer",
    page_icon="😊",
    layout="wide"
)

# ── Global CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 48px 40px;
    color: white;
    margin-bottom: 32px;
    text-align: center;
}
.hero h1 { font-size: 2.6rem; font-weight: 700; margin: 0 0 8px 0; color: white; }
.hero p  { font-size: 1.1rem; opacity: 0.9; margin: 0; }

/* ── Section headings ── */
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #374151;
    margin: 28px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #e5e7eb;
}

/* ── Stat cards ── */
.stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
.stat-card {
    flex: 1; min-width: 140px;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.stat-card .emoji { font-size: 2rem; display: block; margin-bottom: 6px; }
.stat-card .label { font-size: 0.78rem; color: #6b7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-card .value { font-size: 1.1rem; font-weight: 700; color: #111827; margin-top: 4px; }

/* ── Info cards (2-col) ── */
.info-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 24px;
    height: 100%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.info-card h3 { font-size: 1rem; font-weight: 600; color: #374151; margin-top: 0; }
.info-card li { color: #4b5563; margin-bottom: 4px; }

/* ── Team card ── */
.team-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 24px 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.team-row { display: flex; gap: 32px; flex-wrap: wrap; }
.team-member { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.team-member .avatar {
    width: 56px; height: 56px; border-radius: 50%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; color: white;
}
.team-member .name { font-weight: 600; font-size: 0.95rem; color: #111827; }
.team-member .role { font-size: 0.78rem; color: #6b7280; }

/* ── Page header (non-home pages) ── */
.page-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 28px;
}
.page-header h1 { font-size: 1.8rem; font-weight: 700; margin: 0 0 4px 0; color: white; }
.page-header p  { margin: 0; opacity: 0.88; font-size: 0.95rem; }

/* ── Model selector card ── */
.model-selector-label {
    font-size: 0.85rem; font-weight: 600;
    color: #374151; margin-bottom: 4px;
    text-transform: uppercase; letter-spacing: 0.05em;
}

/* ── Result card ── */
.result-card {
    border-radius: 16px;
    padding: 32px 36px;
    margin: 24px 0 16px 0;
    color: white;
    display: flex;
    align-items: center;
    gap: 24px;
}
.result-emoji { font-size: 3.5rem; line-height: 1; }
.result-text  { flex: 1; }
.result-label { font-size: 0.85rem; font-weight: 600; opacity: 0.85; text-transform: uppercase; letter-spacing: 0.08em; }
.result-emotion { font-size: 2.4rem; font-weight: 700; margin: 2px 0 8px 0; }
.result-conf  { font-size: 1rem; opacity: 0.9; }

/* ── Confidence bar (custom) ── */
.conf-bar-wrap { background: rgba(255,255,255,0.25); border-radius: 99px; height: 10px; margin-top: 10px; }
.conf-bar-fill { height: 10px; border-radius: 99px; background: rgba(255,255,255,0.9); }

/* ── Key words ── */
.kw-box {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 16px;
}
.kw-box h4 { margin: 0 0 10px 0; font-size: 0.9rem; color: #374151; font-weight: 600; }
.kw-chip {
    display: inline-block;
    background: #ede9fe;
    color: #5b21b6;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 3px 4px 3px 0;
}

/* ── Model info cards ── */
.model-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.model-card-header {
    display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
}
.model-badge {
    font-size: 0.72rem; font-weight: 600;
    padding: 3px 10px; border-radius: 99px;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.badge-best  { background: #d1fae5; color: #065f46; }
.badge-good  { background: #dbeafe; color: #1e40af; }
.badge-basic { background: #f3f4f6; color: #374151; }
.model-card h3 { margin: 0; font-size: 1rem; font-weight: 600; color: #111827; }
.model-acc { font-size: 1.5rem; font-weight: 700; color: #667eea; }
.model-f1  { font-size: 0.85rem; color: #6b7280; margin-top: 2px; }
.model-desc { font-size: 0.88rem; color: #4b5563; margin-top: 10px; line-height: 1.6; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f7ff 0%, #ffffff 100%);
    border-right: 1px solid #e5e7eb;
}
.sidebar-brand {
    text-align: center;
    padding: 8px 0 16px 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 16px;
}
.sidebar-brand .icon { font-size: 2.4rem; }
.sidebar-brand h2 { font-size: 1.05rem; font-weight: 700; color: #111827; margin: 4px 0 2px 0; }
.sidebar-brand p  { font-size: 0.78rem; color: #6b7280; margin: 0; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #e5e7eb; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

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
    'joy':      '#F59E0B',
    'sadness':  '#3B82F6',
    'anger':    '#EF4444',
    'fear':     '#8B5CF6',
    'love':     '#EC4899',
    'surprise': '#F97316',
}

# ── Preprocessing ─────────────────────────────────────────────
@st.cache_resource
def load_nlp_tools():
    if stopwords is None or WordNetLemmatizer is None:
        class _DummyLem:
            def lemmatize(self, w): return w
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

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="icon">😊</div>
    <h2>Emotion Analyzer</h2>
    <p>SAIA 2163 — NLP Final Project</p>
    <p style="margin-top:4px;font-size:0.76rem;color:#9ca3af;">Syiqin · Ain · Saf</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Text Analyzer", "📊 Data Explorer",
     "📈 Visualizations", "🤖 Model Info"],
    label_visibility="collapsed"
)

# ══════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>😊 Social Media Emotion Analyzer</h1>
        <p>Detect emotions in text using Machine Learning &amp; Natural Language Processing</p>
    </div>
    """, unsafe_allow_html=True)

    # Emotion stat cards
    st.markdown('<div class="section-title">🎭 Emotions We Detect</div>', unsafe_allow_html=True)
    cards_html = '<div class="stat-row">'
    for emo, emoji in EMOJI_MAP.items():
        color = COLOR_MAP[emo]
        cards_html += f"""
        <div class="stat-card">
            <span class="emoji">{emoji}</span>
            <div class="label">{emo.upper()}</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # How to use + Dataset
    st.markdown('<div class="section-title">📖 About This App</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🚀 How to Use</h3>
            <ol style="color:#4b5563;padding-left:20px;margin:0;line-height:1.9;">
                <li>Go to <strong>Text Analyzer</strong></li>
                <li>Choose a model (we recommend DistilBERT)</li>
                <li>Click an example or type your own text</li>
                <li>Click <strong>Analyze</strong> and see results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>📦 Dataset</h3>
            <ul style="color:#4b5563;padding-left:20px;margin:0;line-height:1.9;">
                <li><strong>Source:</strong> DAIR-AI Emotion (Twitter)</li>
                <li><strong>Size:</strong> 20,000 samples</li>
                <li><strong>Classes:</strong> 6 emotions</li>
                <li><strong>Best accuracy:</strong> 93.7% (DistilBERT)</li>
                <li><strong>Split:</strong> 80% train / 20% test</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)

    # Team
    st.markdown('<div class="section-title">👥 Team</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="team-card">
        <div class="team-row">
            <div class="team-member">
                <div class="avatar">S</div>
                <div class="name">Syiqin</div>
                <div class="role">Dataset &amp; Streamlit App</div>
            </div>
            <div class="team-member">
                <div class="avatar">A</div>
                <div class="name">Ain</div>
                <div class="role">NLP Pipeline &amp; Models</div>
            </div>
            <div class="team-member">
                <div class="avatar">S</div>
                <div class="name">Saf</div>
                <div class="role">Analysis &amp; Integration</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2: TEXT ANALYZER
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Text Analyzer":

    st.markdown("""
    <div class="page-header">
        <h1>🔍 Text Emotion Analyzer</h1>
        <p>Enter any social media text and detect its emotion instantly</p>
    </div>
    """, unsafe_allow_html=True)

    stop_words, lemmatizer = load_nlp_tools()
    tfidf, svm = load_classical_model()
    bert_pipe  = load_bert_model()
    w2v_model, w2v_lr = load_w2v_model()

    # Model selector
    st.markdown('<div class="model-selector-label">Choose Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Choose model",
        ["DistilBERT (Best — 93.7%)", "LinearSVC / SVM (90.6%)", "Logistic Regression + Word2Vec (44.7%)"],
        label_visibility="collapsed"
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

    st.markdown('<div class="section-title">💡 Try an Example — click to fill</div>', unsafe_allow_html=True)
    ex_cols = st.columns(3)
    for i, (label, text) in enumerate(EXAMPLES.items()):
        with ex_cols[i % 3]:
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state['text_input'] = text
                st.rerun()

    st.markdown('<div class="section-title">✏️ Your Text</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        "Enter text here:",
        placeholder="Type or paste social media text here...",
        height=120,
        key="text_input",
        label_visibility="collapsed"
    )

    analyze_clicked = st.button("🔍 Analyze Emotion", type="primary", use_container_width=False)

    if analyze_clicked:
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

            # ── Result card ──
            emoji = EMOJI_MAP.get(predicted, '❓')
            color = COLOR_MAP.get(predicted, '#667eea')
            fill_pct = int(confidence * 100)

            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, {color}dd, {color}99);">
                <div class="result-emoji">{emoji}</div>
                <div class="result-text">
                    <div class="result-label">Detected Emotion</div>
                    <div class="result-emotion">{predicted.upper()}</div>
                    <div class="result-conf">Confidence: {confidence*100:.1f}%</div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar-fill" style="width:{fill_pct}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence chart
            st.markdown('<div class="section-title">📊 Confidence for Each Emotion</div>', unsafe_allow_html=True)
            if pd is not None and plt is not None and np is not None:
                score_df = pd.DataFrame({
                    'Emotion': list(scores.keys()),
                    'Confidence': list(scores.values())
                }).sort_values('Confidence', ascending=True)

                fig, ax = plt.subplots(figsize=(8, 3.5))
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#f9fafb')
                bars = ax.barh(
                    score_df['Emotion'], score_df['Confidence'],
                    color=[COLOR_MAP.get(e, '#888') for e in score_df['Emotion']],
                    height=0.55, edgecolor='none'
                )
                ax.set_xlim(0, 1)
                ax.set_xlabel('Confidence Score', fontsize=10, color='#6b7280')
                ax.tick_params(colors='#374151', labelsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#e5e7eb')
                ax.spines['bottom'].set_color('#e5e7eb')
                for bar, val in zip(bars, score_df['Confidence']):
                    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{val:.3f}', va='center', fontsize=9, color='#374151')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.write(scores)

            # Key words
            clean_for_words = preprocess(user_input, stop_words, lemmatizer)
            words_in_vocab = []
            if "SVM" in model_choice and tfidf is not None:
                words_in_vocab = [w for w in clean_for_words.split() if w in tfidf.vocabulary_]
            elif "Word2Vec" in model_choice and w2v_model is not None:
                words_in_vocab = [w for w in clean_for_words.split() if w in w2v_model.wv]

            if words_in_vocab:
                chips = "".join([f'<span class="kw-chip">{w}</span>' for w in words_in_vocab])
                st.markdown(f"""
                <div class="kw-box">
                    <h4>🔑 Key words that influenced this prediction</h4>
                    {chips}
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3: DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":

    st.markdown("""
    <div class="page-header">
        <h1>📊 Data Explorer</h1>
        <p>Browse and explore the DAIR-AI Emotion dataset used to train the models</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.error("Dataset not found. Run the notebook first to generate data/emotion_dataset_processed.csv")
        st.stop()

    # Top metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    with col2:
        st.metric("Emotions", df['emotion'].nunique())
    with col3:
        st.metric("Avg Words / Sample", f"{df['text'].apply(lambda x: len(str(x).split())).mean():.1f}")

    st.markdown('<div class="section-title">🗂 Sample Data</div>', unsafe_allow_html=True)
    st.dataframe(df[['text', 'emotion', 'clean_text']].sample(10, random_state=42), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📈 Class Distribution</div>', unsafe_allow_html=True)
        counts = df['emotion'].value_counts().reset_index()
        counts.columns = ['Emotion', 'Count']
        st.dataframe(counts, use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">📏 Text Length Stats (words)</div>', unsafe_allow_html=True)
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
        st.dataframe(df.groupby('emotion')['word_count']
                     .describe()[['mean','min','max']].round(1), use_container_width=True)

    st.markdown('<div class="section-title">📊 Distribution Chart</div>', unsafe_allow_html=True)
    if plt is not None and sns is not None:
        emotion_counts = df['emotion'].value_counts()
        palette = [COLOR_MAP.get(e, '#888') for e in emotion_counts.index]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#ffffff')
        for ax in axes:
            ax.set_facecolor('#f9fafb')

        axes[0].bar(emotion_counts.index, emotion_counts.values, color=palette, edgecolor='none')
        axes[0].set_title('Count per Emotion', fontsize=11, color='#374151')
        axes[0].set_xlabel('Emotion', color='#6b7280')
        axes[0].set_ylabel('Count', color='#6b7280')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].tick_params(colors='#374151')

        axes[1].pie(emotion_counts.values, labels=emotion_counts.index,
                    autopct='%1.1f%%', colors=palette, startangle=90)
        axes[1].set_title('Distribution (%)', fontsize=11, color='#374151')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info('Plotting requires matplotlib and seaborn.')

# ══════════════════════════════════════════════════════════════
# PAGE 4: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":

    st.markdown("""
    <div class="page-header">
        <h1>📈 Visualizations</h1>
        <p>Charts and analysis generated from the notebook pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    viz_map = {
        "Class Distribution":        "visualizations/class_distribution.png",
        "Word Cloud (All Emotions)":  "visualizations/wordcloud_all.png",
        "Word Clouds per Emotion":    "visualizations/wordclouds_per_emotion.png",
        "Top 20 TF-IDF Words":        "visualizations/top20_tfidf_words.png",
        "Model Comparison":           "visualizations/model_comparison.png",
        "Confusion Matrices":         "visualizations/confusion_matrices.png",
        "Text Length Distribution":   "visualizations/text_length_distribution.png",
    }

    for title, path in viz_map.items():
        if os.path.exists(path):
            st.markdown(f'<div class="section-title">📌 {title}</div>', unsafe_allow_html=True)
            st.image(path, use_container_width=True)
        else:
            st.info(f"**{title}** — not generated yet (run the notebook first)")

# ══════════════════════════════════════════════════════════════
# PAGE 5: MODEL INFO
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Model Info":

    st.markdown("""
    <div class="page-header">
        <h1>🤖 Model Information</h1>
        <p>Architecture, performance, and feature extraction methods explained</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🏆 Performance Summary</div>', unsafe_allow_html=True)

    # Model cards
    models_info = [
        {
            "name": "DistilBERT (fine-tuned)",
            "feature": "Raw text (Transformer)",
            "accuracy": 0.9370, "f1": 0.9111,
            "badge": "badge-best", "badge_label": "⭐ Best Model",
            "desc": "Transformer with 6 attention layers and 66M parameters. Pre-trained on BookCorpus + Wikipedia, fine-tuned for 4 epochs on 16,000 training samples. Understands full sentence context and handles nuance extremely well."
        },
        {
            "name": "LinearSVC / SVM",
            "feature": "TF-IDF (10k features, bigrams)",
            "accuracy": 0.9055, "f1": 0.8704,
            "badge": "badge-good", "badge_label": "✅ Best Classical",
            "desc": "Support Vector Machine with linear kernel. Uses TF-IDF features with unigrams + bigrams and balanced class weighting. Fast, interpretable, and achieves strong accuracy without a GPU."
        },
        {
            "name": "Logistic Regression",
            "feature": "TF-IDF (10k features, bigrams)",
            "accuracy": 0.9000, "f1": 0.8685,
            "badge": "badge-good", "badge_label": "✅ Strong",
            "desc": "Classic probabilistic classifier with L2 regularisation. Very similar performance to SVM on this dataset."
        },
        {
            "name": "Naive Bayes",
            "feature": "TF-IDF (10k features)",
            "accuracy": 0.7578, "f1": 0.5754,
            "badge": "badge-basic", "badge_label": "📊 Baseline",
            "desc": "Multinomial Naive Bayes. Simple and fast but assumes word independence — this hurts on emotional text where phrase context matters."
        },
        {
            "name": "Logistic Regression + Word2Vec",
            "feature": "Word2Vec (averaged vectors)",
            "accuracy": 0.4470, "f1": 0.3940,
            "badge": "badge-basic", "badge_label": "📊 Comparison",
            "desc": "Logistic Regression using averaged Word2Vec embeddings. Low accuracy shows that averaging vectors loses word order and context — 'not happy' and 'happy' get similar vectors. Included to show why feature choice matters."
        },
    ]

    for m in models_info:
        st.markdown(f"""
        <div class="model-card">
            <div class="model-card-header">
                <div>
                    <h3>{m['name']}</h3>
                    <div style="font-size:0.82rem;color:#6b7280;margin-top:2px;">Feature: {m['feature']}</div>
                </div>
                <span class="model-badge {m['badge']}">{m['badge_label']}</span>
            </div>
            <div style="display:flex;gap:24px;align-items:flex-end;margin-bottom:8px;">
                <div>
                    <div style="font-size:0.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;">Accuracy</div>
                    <div class="model-acc">{m['accuracy']:.1%}</div>
                </div>
                <div>
                    <div style="font-size:0.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;">Macro-F1</div>
                    <div class="model-acc" style="font-size:1.2rem;">{m['f1']:.1%}</div>
                </div>
            </div>
            <div class="model-desc">{m['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">⚙️ Training Setup</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Dataset & Split</h3>
            <ul style="color:#4b5563;padding-left:20px;margin:0;line-height:1.9;">
                <li>DAIR-AI Emotion — 20,000 Twitter samples</li>
                <li>80% train / 20% test (stratified)</li>
                <li>random_state = 42</li>
                <li>6 emotion classes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Hardware Support</h3>
            <ul style="color:#4b5563;padding-left:20px;margin:0;line-height:1.9;">
                <li>🍎 Apple Silicon — MPS acceleration</li>
                <li>🟢 NVIDIA GPU — CUDA acceleration</li>
                <li>💻 CPU — fallback (slower)</li>
                <li>Auto-detected at runtime</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
