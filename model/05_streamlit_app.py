import streamlit as st
import pickle
import numpy as np
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from pathlib import Path

# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="ReviewLens", page_icon="🔍", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
DATA_DIR = BASE_DIR / "data"


@st.cache_resource
def load_models():
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_DIR / "nb_model.pkl", "rb") as f:
        nb_model = pickle.load(f)
    with open(MODEL_DIR / "svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open(MODEL_DIR / "log_odds.pkl", "rb") as f:
        log_odds = pickle.load(f)
    with open(MODEL_DIR / "nb_results.pkl", "rb") as f:
        nb_results = pickle.load(f)
    with open(MODEL_DIR / "svm_results.pkl", "rb") as f:
        svm_results = pickle.load(f)
    with open(MODEL_DIR / "rf_results.pkl", "rb") as f:
        rf_results = pickle.load(f)
    y_test = np.load(DATA_DIR / "y_test.npy")

    return vectorizer, nb_model, svm_model, log_odds, nb_results, svm_results, rf_results, y_test

vectorizer, nb_model, svm_model, log_odds, nb_results, svm_results, rf_results, y_test = load_models()
feature_names = vectorizer.get_feature_names_out()


@st.cache_data
def get_stopwords():
    negation = {"no", "not", "nor", "never", "neither", "without", "nobody", "nothing", "nowhere", "hardly", "barely"}
    return set(stopwords.words('english')) - negation

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = get_stopwords()
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens), tokens

def predict_review(raw_text, model):
    clean, tokens = clean_text(raw_text)
    if not clean.strip():
        return None, None, {}, clean
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = proba[prediction]
    vocab = vectorizer.vocabulary_
    word_scores = {}
    for token in set(tokens):
        if token in vocab:
            idx = vocab[token]
            if vec[0, idx] > 0:
                word_scores[token] = log_odds[idx]
    return int(prediction), float(confidence), word_scores, clean

def highlight_html(raw_text, word_scores, verdict_color):
    lemmatizer = WordNetLemmatizer()
    stop_words = get_stopwords()
    max_abs = max(abs(v) for v in word_scores.values()) if word_scores else 1
    display_tokens = re.sub(r'[^\w\s]', '', raw_text.lower()).split()
    html_parts = []
    for word in display_tokens:
        lemma = lemmatizer.lemmatize(word)
        if lemma in word_scores:
            score = word_scores[lemma]
            intensity = min(abs(score) / max_abs, 1.0)
            if score > 0.05:
                r, g, b = 255, int(255*(1-intensity*0.85)), int(255*(1-intensity*0.85))
                title = f"FAKE signal: {score:+.3f}"
            elif score < -0.05:
                r, g, b = int(255*(1-intensity*0.85)), int(255*(1-intensity*0.85)), 255
                title = f"REAL signal: {score:+.3f}"
            else:
                html_parts.append(word)
                continue
            html_parts.append(
                f'<span style="background-color:rgb({r},{g},{b}); padding:2px 5px; '
                f'border-radius:3px; margin:1px; cursor:help;" '
                f'title="{title}">{word}</span>'
            )
        else:
            html_parts.append(word)
    return ' '.join(html_parts)


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.radio( "Choose classifier:", ["SVM (recommended)", "Naive Bayes"], help="SVM: higher overall accuracy. NB: better at catching fakes.")
    st.markdown("---")
    st.header("📊 Model Performance")
    perf_data = {
        "Metric": ["Accuracy", "AUC", "Sensitivity", "Specificity"], "SVM": ["91.56%", "0.9662", "91.25%", "91.87%"],
        "Naive Bayes": ["90.62%", "0.9648", "93.75%", "87.50%"],
        "Rnd Forest": ["88.12%", "0.9518", "87.50%", "88.75%"],
    }
    st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)


tab1, tab2 = st.tabs(["🔍 Detector", "📊 Model Comparison"])


# Tab 1 — Detector
with tab1:
    st.title("🔍 ReviewLens")
    st.markdown("*Paste any review to analyze it.*")

    example_fake = """This hotel was absolutely amazing! The staff was incredibly wonderful and
the rooms were perfect and luxurious. Best hotel in Chicago by far! I will definitely
come back and recommend to all my friends and family. Outstanding experience overall,
truly a world class luxury destination that exceeded every expectation!"""

    example_real = """Stayed for 3 nights in October. Room on 8th floor was small but clean.
Bathroom had a broken towel rack which maintenance fixed next day. Breakfast buffet
was decent, eggs were cold though. Location is great for walking to Millennium Park.
Booked through Priceline, got a decent rate. Would probably stay again for the price."""

    col1, col2 = st.columns(2)
    if col1.button("🚨 Load Suspicious Review", use_container_width=True):
        st.session_state['review_input'] = example_fake
    if col2.button("✅ Load Genuine Review", use_container_width=True):
        st.session_state['review_input'] = example_real

    review_text = st.text_area("Paste your review here:", value=st.session_state.get('review_input', ''),height=180, 
                               placeholder="Type or paste a hotel review (minimum 10 words)...")

    analyze_clicked = st.button("🔍 Analyze Review", type="primary", use_container_width=True)

    if analyze_clicked:
        word_count = len(review_text.strip().split())
        if not review_text.strip():
            st.warning("⚠️ Please enter a review first.")
        elif word_count < 10:
            st.warning(f"⚠️ Review too short ({word_count} words). Please enter at least 10 words for a reliable prediction.")
        elif word_count > 600:
            st.warning("⚠️ Review is very long. Truncating to first 600 words for analysis.")
            review_text = ' '.join(review_text.split()[:600])
        else:
            model = svm_model if "SVM" in model_choice else nb_model
            with st.spinner("Analyzing review..."):
                pred, conf, word_scores, clean = predict_review(review_text, model)

            if pred is None:
                st.error("Could not extract meaningful features from this review. Try a longer, more descriptive review.")
            else:
                st.markdown("---")
                if pred == 1:
                    st.error("## 🚨 FAKE Review Detected")
                    verdict_color = "#ff4b4b"
                else:
                    st.success("## ✅ GENUINE Review")
                    verdict_color = "#21c354"

                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.metric("Confidence Score", f"{conf*100:.1f}%")
                    st.progress(float(conf))

                st.markdown("---")
                st.subheader("📝 Word-Level Analysis")
                st.caption("🔴 Red = fake-signalling  |  🔵 Blue = genuine-signalling  |  Hover over a word to see its score")

                highlighted = highlight_html(review_text, word_scores, verdict_color)
                st.markdown(
                    f'<div style="background:#1e1e1e; padding:20px; border-radius:10px; '
                    f'font-size:16px; line-height:2.4; border-left:4px solid {verdict_color};">'
                    f'{highlighted}</div>',
                    unsafe_allow_html=True)

                if word_scores:
                    st.markdown("---")
                    col_f, col_r = st.columns(2)
                    sorted_scores = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
                    fake_words = [(w,round(s, 3)) for w, s in sorted_scores if s > 0.05][:8]
                    real_words = [(w, round(s, 3)) for w, s in sorted_scores if s < -0.05][-8:]

                    with col_f:
                        st.subheader("🚨 Fake signals")
                        if fake_words:
                            st.dataframe(pd.DataFrame(fake_words, columns=["Word", "Score"]),
                                         hide_index=True, use_container_width=True)
                        else:
                            st.info("No strong fake signals found.")

                    with col_r:
                        st.subheader("✅ Real signals")
                        if real_words:
                            st.dataframe(pd.DataFrame(real_words, columns=["Word", "Score"]),
                                         hide_index=True, use_container_width=True)
                        else:
                            st.info("No strong real signals found.")

                with st.expander("🔬 Transparency — cleaned tokens"):
                    st.code(clean)
                    st.caption(f"Model: {model_choice} | Vocabulary matches: {len(word_scores)} words")


# Tab 2 — Model Comparison
with tab2:
    st.title("📊 Model Comparison")

    st.subheader("ROC Curves")
    fig_roc, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nb_results['fpr'], nb_results['tpr'],
            color='tomato', lw=2, label=f"Naive Bayes   (AUC={nb_results['auc']:.4f})")
    ax.plot(svm_results['fpr'], svm_results['tpr'],
            color='steelblue', lw=2, label=f"SVM           (AUC={svm_results['auc']:.4f})")
    ax.plot(rf_results['fpr'], rf_results['tpr'],
            color='seagreen', lw=2, label=f"Random Forest (AUC={rf_results['auc']:.4f})")
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random (AUC=0.5)')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — All Models")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_roc)
    plt.close()

    st.subheader("Confusion Matrices")
    fig_cm, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, res, cmap) in zip(axes, [("Naive Bayes", nb_results, "Reds"), ("SVM", svm_results, "Blues"),("Random Forest", rf_results, "Greens"),]):
        cm = res['cm']
        tn, fp, fn, tp = cm.ravel()
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=['Pred Real', 'Pred Fake'], yticklabels=['Act Real', 'Act Fake'], 
                    ax=ax, linewidths=0.5, annot_kws={"size": 13})
        ax.set_title(f"{name}\nAcc={res['accuracy']*100:.1f}%  FP={fp}  FN={fn}")
    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close()

    st.subheader("Summary Table")
    summary = pd.DataFrame({
        "Model": ["Naive Bayes", "SVM", "Random Forest"],

        "Accuracy": [f"{nb_results['accuracy']*100:.2f}%", f"{svm_results['accuracy']*100:.2f}%", f"{rf_results['accuracy']*100:.2f}%"],

        "AUC": [f"{nb_results['auc']:.4f}", f"{svm_results['auc']:.4f}", f"{rf_results['auc']:.4f}"],

        "Sensitivity": [f"{nb_results['sensitivity']:.4f}", f"{svm_results['sensitivity']:.4f}", f"{rf_results['sensitivity']:.4f}"],

        "Specificity": [f"{nb_results['specificity']:.4f}", f"{svm_results['specificity']:.4f}", f"{rf_results['specificity']:.4f}"],

        "Top Performance in": ["Sensitivity", "Accuracy + AUC", "—"],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)