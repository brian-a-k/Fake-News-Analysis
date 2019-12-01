"""
Main script for cleaning (Lemmatization) the merged real and fake news DataSets
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


def nlp_lemmatization(corpus: np.ndarray) -> np.ndarray:
    nlp = spacy.load('en_core_web_lg')
    pipeline = nlp.pipe(corpus)
    for idx, doc in enumerate(pipeline):
        lemma_tokens = [token.lemma_.strip().lower() for token in doc if token.is_stop is False
                        and token.text not in punctuation and token.lemma_ != '-PRON-']
        corpus[idx] = ' '.join(lemma_tokens)
    return corpus


def main(merged_news_df: pd.DataFrame) -> pd.DataFrame:
    # Keep just the news content columns and drop any null Text columns
    news_content = merged_news_df[['title', 'text', 'type']]

    # Basic text cleaning of the title and text columns
    news_content = clean_text_column(news_content, 'title')
    news_content = clean_text_column(news_content, 'text')

    # Create our NLP tokenized column for title
    title_array = pd.Series(news_content.title).to_numpy(dtype=object, copy=True)
    title_tokens = nlp_lemmatization(corpus=title_array)
    news_content['tokenized_headline'] = title_tokens

    # Create our NLP tokenized column for text
    text_array = pd.Series(news_content.text).to_numpy(dtype=object, copy=True)
    text_tokens = nlp_lemmatization(corpus=text_array)
    news_content['tokenized_content'] = text_tokens

    # Return cleaned, nlp processed DataFrame
    return news_content


if __name__ == '__main__':
    # EDIT for the local fake and real news merged data
    input_path = ''
    # EDIT for your local save file path
    save_path = ''

    # Processing
    news_data = pd.read_csv(input_path)
    nlp_lemma_df = main(news_data)
    nlp_lemma_df.to_csv(path_or_buf=save_path, header=True, index=None)
