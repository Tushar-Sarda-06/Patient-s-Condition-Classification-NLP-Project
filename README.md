Patient's Condition Classification | NLP Project

_A machine learning pipeline that classifies patient medication conditions from 70K+ drug reviews using advanced NLP techniques and ensemble learning._

**Final Accuracy: 93.66% | Weighted Avg Precision: 0.937 | Weighted Avg Recall: 0.937**


**📋 Project Overview**

This project predicts one of 9 medical conditions from patient drug reviews by:


Data-driven scoping — reduced 885 conditions to 9 optimal conditions using statistical cliff analysis
Advanced feature engineering — engineered TF-IDF trigrams with POS-aware lemmatization
Ensemble modeling — trained RandomForest, XGBoost, and LightGBM with rigorous hyperparameter tuning
Production-ready evaluation — generated confusion matrices, per-class metrics, and feature importance analysis



🎯 Problem Statement

The dataset contains 161K+ patient reviews across 885+ medical conditions. The challenge:


Massive class imbalance — Birth Control dominates with 28K+ reviews vs ADHD with only 3.3K
Condition overlap — Depression, Anxiety, Bipolar Disorder share 63–75% vocabulary
Noisy data — 899 null labels, spelling variations, medical jargon
Computational efficiency — process large sparse text data without transformers



✨ Key Features

Data Preprocessing


Optimal condition selection: Used statistical cliff-detection analysis to identify natural break at 24.5% drop (rank 10→11)
Class imbalance handling: 8.5x imbalance managed via class_weight='balanced_subsample'
POS-aware lemmatization: WordNetLemmatizer with POS tagging for semantically accurate token normalization
Negation preservation: Kept "not", "no", "never" in stopwords to preserve medical signal ("not working" vs "working")
HTML stripping & tokenization: BeautifulSoup + NLTK word_tokenize for clean text


Feature Engineering


TF-IDF trigrams (1-3 grams)

80,000 vocabulary size (max_features)
Sublinear scaling (sublinear_tf=True) — applies log normalization to reduce term-frequency bias
min_df=2 to remove rare typos
max_df=0.80 to remove overly frequent words



Empirically validated: Tested TF-IDF vs TF-IDF+Word2Vec vs TF-IDF+FastText

Result: TF-IDF alone outperformed embeddings by 1% — sparse token importance > semantic density for this task





Models & Ensembling

ModelAccuracyNotesRandomForest91.71%OOB Score: 0.918 (unbiased generalization)XGBoost91.54%Shallow trees (max_depth=6) for sparse dataLightGBM93.66%Best single model — leaf-wise growth excels on sparse matricesVoting Ensemble93.67%+0.001% improvement over LightGBM; chose single model for efficiency

Per-Class Performance (LightGBM)

Condition              Precision  Recall  F1-Score  Support
ADHD                   0.9458    0.9272  0.9364    659
Acne                   0.9663    0.9314  0.9485    1,108
Anxiety                0.8710    0.8124  0.8407    1,130
Bipolar Disorder       0.9098    0.8529  0.8805    816
Birth Control          0.9805    0.9913  0.9859    5,732
Depression             0.8265    0.8935  0.8587    1,765
Insomnia               0.8815    0.8640  0.8727    706
Pain                   0.9535    0.9453  0.9494    1,151
Weight Loss            0.9734    0.9707  0.9721    717


🛠️ Tech Stack

Languages: Python 3.8+

Data Processing:


pandas — data manipulation
numpy — numerical operations
beautifulsoup4 — HTML parsing


NLP & Feature Engineering:


nltk — tokenization, POS tagging, lemmatization, stopwords
scikit-learn — TF-IDF vectorization
gensim (optional) — Word2Vec/FastText for comparison


**Machine Learning:**


scikit-learn — RandomForest, preprocessing, metrics
xgboost — XGBoost classifier
lightgbm — LightGBM classifier
scipy.sparse — sparse matrix operations


**Visualization:**


matplotlib, seaborn — plots and heatmaps
wordcloud — word clouds per condition


**📊 Data Insights**

Condition Selection Process


Initial dataset: 161K reviews across 885 conditions
Statistical cliff analysis: Plotted condition frequency distribution
Natural break point: 24.5% drop between rank 10 (ADHD: 3.3K) and rank 11 (Diabetes Type 2: 2.5K)
Final selection: Top 9 conditions = 70K reviews (43.6% of dataset)


**Class Distribution**

Birth Control    28,788 reviews   (40.9%)
Depression        9,069 reviews   (12.9%)
Pain              6,145 reviews   (8.7%)
Anxiety           5,904 reviews   (8.4%)
Acne              5,588 reviews   (7.9%)
Bipolar Disorder  4,224 reviews   (6.0%)
Insomnia          3,673 reviews   (5.2%)
Weight Loss       3,609 reviews   (5.1%)
ADHD              3,383 reviews   (4.8%)

**Imbalance ratio: 8.5x (manageable with class weighting)**


🔍 Key Insights

Why TF-IDF Won Over Embeddings


TF-IDF alone: 91.77% accuracy
TF-IDF + Word2Vec: 90.69% accuracy (-1.09%)
TF-IDF + FastText: 90.72% accuracy (-1.06%)


Finding: On this dataset, conditions are separated by distinct vocabulary, not semantic similarity. Conditions have fundamentally different tokens (e.g., "period" → Birth Control, "focus" → ADHD). Embeddings add noise without signal.

Why LightGBM Won


Leaf-wise tree growth — more efficient on 80K sparse features than level-wise (XGBoost)
Native sparse matrix support — no dense conversion overhead
Categorical feature optimization — automatically handles categorical splits
Final accuracy: 93.66% (0.95% above RandomForest, 2.1% above XGBoost)


Challenging Pairs

Highest vocabulary overlap (hardest to distinguish):


Depression ↔ Anxiety: 75.4% overlap
Depression ↔ Bipolar: 69.5% overlap
Anxiety ↔ Bipolar: 63.9% overlap


Strategy used: POS-aware lemmatization + negation preservation ("not depressed" vs "depressed")


📈 Results Summary

MetricValueTest Accuracy93.66%Weighted Avg Precision0.9371Weighted Avg Recall0.9366Per-class Precision Range0.83–0.98Per-class Recall Range0.81–0.99Best Single Class (Birth Control)Precision: 0.98, Recall: 0.99Hardest Class (Anxiety)Precision: 0.87, Recall: 0.81


🧪 Methodology

1. Data Cleaning


Dropped 899 rows with null condition labels
Removed 410 rows with <10 word reviews (insufficient signal)
Final dataset: 68,919 rows → 55,135 train, 13,784 test


2. Text Preprocessing

Raw Review → HTML stripping → Lowercasing → Negation gluing 
→ Tokenization → Stopword removal → POS tagging → Lemmatization 
→ Cleaned review

3. Feature Engineering


TF-IDF Vectorizer with:

ngram_range=(1,3) — captures "does not work", "no side effects"
sublinear_tf=True — log normalization
max_features=80000 — large vocabulary
min_df=2 — removes rare tokens


4. Model Training

RandomForest
XGBoost
LightGBM
Soft Voting Ensemble


5. Evaluation

Confusion matrix heatmap
Per-class classification report
Feature importance ranking (Top 20)
Cross-validation scores
