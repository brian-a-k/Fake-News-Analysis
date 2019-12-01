"""
Main script for cleaning the 'Getting real about fake news' DataSet
"""
from string import punctuation

import numpy as np
import pandas as pd
import spacy


# Fills null string columns with a default string value
def fill_null_string_columns(data: pd.DataFrame, fill_dict: dict):
    return data.fillna(value=fill_dict)


# Removes all characters that are NOT: (letters numbers . ? , !) and any duplicate whitespace
def clean_text_column(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    data[column_name] = data[column_name].str.replace('[^0-9a-zA-Z .!?,]', '', regex=True).str.split().str.join(' ')
    return data


# NLP processing for NERs and Noun tokens
def nlp_tokenize(corpus: np.ndarray) -> np.ndarray:
    nlp = spacy.load('en_core_web_md')
    pipeline = nlp.pipe(corpus)
    ner_types = ['PERSON', 'NORP', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']
    noun_types = ['PROPN', 'NOUN']

    for idx, doc in enumerate(pipeline):
        # Get all nouns and proper nouns
        noun_tokens = [token for token in doc if token.pos_ in noun_types]

        # Get entities from nouns
        entity_tokens = [token for token in noun_tokens if token.ent_type_ in ner_types]

        # Lemmatization of text (shortens it)
        lemma_tokens = [token.lemma_.strip().lower() for token in entity_tokens if token.is_stop is False
                        and token.text not in punctuation]

        # Set new array value
        if len(lemma_tokens) > 0:
            corpus[idx] = ' '.join(lemma_tokens)
        else:
            corpus[idx] = 'notokes'  # default for no found tokens in the text (removed within STOPWORDS)
    return corpus


def main(fake_news: pd.DataFrame) -> pd.DataFrame:
    # Keep just English articles
    fake_news = fake_news[fake_news.language == 'english']

    # Keep just the news content columns and drop any null Text columns
    fake_news_content = fake_news[['title', 'text', 'type']]
    fake_news_content = fake_news_content.dropna(subset=['text'])
    fake_news_content = fake_news_content.reset_index(drop=True)

    # Basic text cleaning of the title and text columns
    fake_news_content = clean_text_column(fake_news_content, 'title')
    fake_news_content = clean_text_column(fake_news_content, 'text')

    # fill any blank titles
    title_fill = {'title': 'default_title'}
    fake_news_content = fill_null_string_columns(fake_news_content, fill_dict=title_fill)

    # Create our NLP tokenized column for title
    title_array = pd.Series(fake_news_content.title).to_numpy(dtype=object, copy=True)
    title_tokens = nlp_tokenize(corpus=title_array)
    fake_news_content['tokenized_headline'] = title_tokens

    # Create our NLP tokenized column for text
    text_array = pd.Series(fake_news_content.text).to_numpy(dtype=object, copy=True)
    text_tokens = nlp_tokenize(corpus=text_array)
    fake_news_content['tokenized_content'] = text_tokens

    # Return cleaned, nlp processed DataFrame
    return fake_news_content


if __name__ == '__main__':
    # EDIT for your local path to the raw raw_data
    # Make sure this is just the raw .csv file from: https://www.kaggle.com/mrisdal/fake-news
    raw_data = pd.read_csv('/Users/briankalinowski/Desktop/Data/Kaggle/fake_news.csv')

    # Processing
    nlp_processed = main(raw_data)

    # EDIT for your local save file path
    save_path = '/Users/briankalinowski/Desktop/Data/fake_news_nlp_content.csv'
    nlp_processed.to_csv(path_or_buf=save_path, header=True, index=None)
