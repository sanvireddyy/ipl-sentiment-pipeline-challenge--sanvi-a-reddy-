# ipl-sentiment-pipeline-challenge--sanvi-a-reddy-
# IPL Sentiment Classification – Pipeline Selection Project

##  Project Overview

This project builds and evaluates multiple machine learning pipelines for IPL sentiment classification using social media text. The task is to classify IPL-related text into:

- Positive
- Negative
- Neutral

The dataset consists of short, noisy, and informal social media text such as tweets and comments about IPL matches, teams, and players.

The main challenges include:
- Slang and abbreviations (e.g., "RCB", "MI", "kkr")
- Spelling variations and exaggerated words (e.g., "goooood", "awwwwsome")
- Hinglish (mix of English and Indian languages)
- Short context-dependent sentences

---

##  Objective

The goal of this project is to:
- Compare multiple text vectorization techniques
- Compare multiple classification algorithms
- Handle noisy real-world social media text
- Balance accuracy, efficiency, and interpretability

---

##  Repository Structure
submission/
│
├── completed_notebook.ipynb
├── submission_manifest.json
├── results_80R.csv
├── requirements.txt


---

##  Methods Used

###  Vectorizers
- CountVectorizer (baseline frequency approach)
- TF-IDF Word N-grams (captures important words and phrases)
- TF-IDF Character N-grams (handles typos, slang, Hinglish)
- HashingVectorizer (scalable representation without vocabulary storage)

###  Classifiers
- Multinomial Naive Bayes (fast baseline model)
- Logistic Regression (strong and interpretable linear model)
- Linear Support Vector Classifier (high-performance margin-based model)
- SGDClassifier (efficient scalable linear learning model)

---

## 🧪 Experimental Design

All vectorizer × classifier combinations were evaluated using:

- Accuracy
- Macro F1-score
- Precision and Recall
- Training time
- Prediction time
- Model size

This ensures fair comparison between simple, efficient, and high-performing models.

---

##  Key Insights

- TF-IDF models perform best for sentiment understanding
- Character n-grams are essential for noisy IPL text
- Linear models generalize well on sparse text data
- Naive Bayes provides a strong and fast baseline

No single model dominates all metrics; trade-offs exist between accuracy and efficiency.

---

## Final Selected Models

### Vectorizers (4)
- CountVectorizer
- TF-IDF Word N-grams
- TF-IDF Character N-grams
- HashingVectorizer

### Classifiers (4)
- Multinomial Naive Bayes
- Logistic Regression
- LinearSVC
- SGDClassifier

---

##  Installation

Install dependencies using:

```bash
pip install -r requirements.txt

 Author
Name: sanvi a reddy 
Team: group 3

 License

For academic submission and educational purposes only.

#  requirements.txt

```text id="req_full"
numpy
pandas
scikit-learn
matplotlib
joblib
