# 🔍 Fake Review Detector

A machine learning project that classifies hotel reviews as **genuine or fake**, explains *why* using word-level highlighting, and serves predictions through a live Streamlit web app.

Built as part of my B.Tech Machine Learning course using the Deceptive Opinion Spam dataset (Ott et al., 2011).

## Results

| Model | Accuracy | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| SVM (recommended)| 91.56%| 0.9662 | 91.25% | 91.87% |
| Naive Bayes | 90.62% | 0.9648 | 93.75% | 87.50% |
| Random Forest | 88.12% | 0.9518 | 87.50% | 88.75% |

SVM wins on overall accuracy and AUC. Naive Bayes catches more fakes (highest sensitivity) and is better if missing a fake review is costly.

---

## How It Works

```
Raw Review Text
      ↓
Text Preprocessing (lowercase, remove stopwords, lemmatize)
      ↓
TF-IDF Vectorization (5000 features, unigrams + bigrams)
      ↓
Trained Classifier (SVM or Naive Bayes)
      ↓
Prediction + Confidence Score
      ↓
Word Highlighting via Naive Bayes Log-Odds
```

**TF-IDF:** Raw word counts are dominated by common words like "hotel" that appear in every review. TF-IDF down-weights these and amplifies rare, discriminating words — the real signal for fake detection.

**WORD HIGHLIGHTING:** The log-odds score from Naive Bayes tells us, for each word, how much it pushes toward FAKE vs REAL. Words like *luxury*, *vacation*, *relax* are strong fake signals. Words like *elevator*, *priceline*, *bathroom* signal a real reviewer who actually stayed there.

---

## Dataset

**Deceptive Opinion Spam Corpus** — Ott, Choi, Cardie & Hancock (2011)

- 1,600 hotel reviews about 20 Chicago hotels
- 800 fake (written by paid Amazon Mechanical Turk workers)
- 800 real (scraped from TripAdvisor)
- Perfectly balanced due to which there was no resampling required
- All reviews are 1-star or 5-star ratings

---

## Project Structure

```
project/
├── raw_data/
│   └── Deceptive-Review-Detection-main/
│       └── Chicago_Hotel_Review/
│           └── Chicago_Hotel_Reviews.csv
│
└── model/
    ├── data/                         
    ├── saved_models/                  
    ├── plots/                         
    ├── 01_eda_and_preprocessing.ipynb
    ├── 02_feature_extraction.ipynb
    ├── 03_model_training.ipynb
    ├── 04_explainability.ipynb
    └── 05_streamlit_app.py
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Shubh0507/Fake-Review-Detector.git
cd Fake-Review-Detector
```

### 2. Create a virtual environment
```bash
# Create a virtual environment 
python -m venv .venv

# Activation
.venv\Scripts\activate

```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebooks in order
Open Jupyter and run each notebook inside the `model/` folder:
```
01_eda_and_preprocessing.ipynb
02_feature_extraction.ipynb
03_model_training.ipynb
04_explainability.ipynb
```

### 5. Launch the Streamlit app
```bash
cd model
streamlit run 05_streamlit_app.py
```

Visit `http://localhost:8501` in your browser.

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
streamlit
fpdf2
jupyter
```

Or install everything at once:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk streamlit fpdf2 jupyter
```

---

## Key Findings

Fake reviews use vague, superlative language: Words like *luxury*, *vacation*, *relax*, *amazing*, *outstanding* were the strongest fake indicators. Paid reviewers who haven't stayed at the hotel compensate with generic praise.

Real reviews have physical details:  Words like *elevator*, *priceline*, *block*, *bathroom*, *bed* signal a reviewer who  actually experienced the hotel.

RF underperformance: Tree ensembles randomly sample ~70 of 5000 features per split. On a 97% sparse TF-IDF matrix, most sampled features are zero due to which the splits become uninformative. NB and SVM use all features simultaneously.

Use of Bigrams: Hotel brand names like *hyatt regency*, *intercontinental chicago* appeared as strong fake indicators. This is because paid reviewers over-mention brand names to sound credible.

---

## Limitations

- Dataset covers Chicago hotels only (2011) and may not generalize across domains
- No behavioral features used (reviewer history, posting frequency, account age)
- TF-IDF ignores word order and context — sarcasm and irony are not handled
- Threshold is fixed at 0.5. Production use would require tuning based on cost tradeoffs

---

## References

Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). *Finding Deceptive Opinion Spam by Any Stretch of the Imagination.* Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics.