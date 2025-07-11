Cleaned 200K+ patient reviews, encoded categorical variables, and normalized features using StandardScaler.
Transformed reviews using NLTK removed stopwords, applied WordNetLemmatizer, extracted features withTF-IDF.
Trained a RandomForestClassifier with hyperparameter tuning via RandomizedSearchCV, achieving 92.4%
accuracy in classifying 7 conditions and an OOB score of 0.92, with precision up to 0.99 and recall up to 1.00 .
Enhanced accuracy to 92.8%using VotingClassifier, ensemble of XGBoost and RandomForest.
