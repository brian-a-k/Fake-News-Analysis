import nltk
import numpy as np
import pandas as pd
from typing import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS


def get_sentiment_vader_scores(data: pd.DataFrame, text_col: str) -> pd.DataFrame:
    # Look for vader_lexicon resource if not found then download
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

    # Sentiment analyzer object
    sentiment_analyser = SentimentIntensityAnalyzer()
    # All documents text
    corpus = [str(doc) for doc in data[text_col]]
    # Get vader polarity scores and create a DataFrame
    vader_df = pd.DataFrame([sentiment_analyser.polarity_scores(txt) for txt in corpus])
    vader_df['sentiment_score'] = vader_df.apply(assign_sentiment_score, axis=1)
    # Concat to original DataFrame
    return pd.concat([data, vader_df], axis=1)


def run_random_forest_grid_search(x_train: pd.DataFrame, y_train: pd.Series, cv_params: dict,
                                  x_test: pd.DataFrame) -> Tuple[Any, Any]:
    # Random Forest Classifier
    random_forest_model = RandomForestClassifier(random_state=21)

    # Random Forest Grid Search CV
    rf_grid_search = GridSearchCV(random_forest_model, cv_params, scoring='accuracy', cv=5, n_jobs=-1, verbose=10)
    rf_grid_search.fit(x_train, y_train)

    # Show best CV parameters and best training score
    print('Best Training Parameters:', rf_grid_search.best_params_)
    print('Best Training Score:', rf_grid_search.best_score_)

    # Return best Random Forest predictions, and probabilities
    return rf_grid_search.predict(x_test), rf_grid_search.predict_proba(x_test)


def get_count_vectorizer_matrix(data: pd.DataFrame, text_col: str) -> np.array:
    # Combine both NLP Libraries stop word lists
    stop_1 = set(stopwords.words('english'))
    stop_2 = set(STOP_WORDS)
    master_stop = list(stop_2.union(stop_1))
    # Create word count vector matrix
    vectorizer = CountVectorizer(min_df=10, stop_words=master_stop)
    content_vectorized = vectorizer.fit_transform(data[text_col])
    print('Count Vectorizer Shape:', content_vectorized.shape, '\n')
    return content_vectorized


def get_lda_transformed_topics(word_vectors: np.array) -> pd.DataFrame:
    # LDA transformer
    lda = LatentDirichletAllocation(n_components=10, max_iter=10, learning_method='online',
                                    verbose=True, n_jobs=-1, random_state=21)

    # fit and transform word matrix to topic probabilities
    lda_topics = lda.fit_transform(word_vectors)
    # Reformat as a DataFrame
    topics_df = pd.DataFrame(lda_topics)
    topics_df.columns = [('topic_' + str(col)) for col in list(topics_df.columns)]
    return topics_df


def format_classification_report(y_test: pd.Series, predictions: pd.Series) -> pd.DataFrame:
    labels = ['fake', 'real']
    return pd.DataFrame(classification_report(y_test, predictions, target_names=labels, output_dict=True)).transpose()


def format_confusion_matrix(y_test: pd.Series, predictions: pd.Series) -> pd.DataFrame:
    conf_matrix = pd.DataFrame(confusion_matrix(y_test, predictions, labels=[0, 1]),
                               index=['True_Fake', 'True_Real'], columns=['Predict_Fake', 'Predict_Real'])

    conf_matrix['True_Totals'] = [sum(row) for row in conf_matrix.values]
    return conf_matrix


def assign_sentiment_score(row):
    if row['compound'] >= 0.05:
        return 1  # positive
    elif row['compound'] <= -0.05:
        return -1  # negative
    else:
        return 0  # neutral
