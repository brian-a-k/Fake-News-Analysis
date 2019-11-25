from typing import *
import pandas as pd
import numpy as np
from collections import defaultdict
import spacy
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# Just formats column names
def clean_columns(raw_cols: List[str]) -> List[str]:
    return [name.lower().replace(' ', '_') for name in raw_cols]


def fill_null_string_columns(data: pd.DataFrame, fill_dict: dict):
    return data.fillna(value=fill_dict)


# Removes all characters that are NOT: (letters numbers . ? , !)
def clean_text_column(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    # raw_data[column_name] = raw_data[column_name].str.replace('[^0-9a-zA-Z’ .!?,]', '', regex=True)
    data[column_name] = data[column_name].str.replace('[^0-9a-zA-Z .!?,]', '', regex=True).str.split().str.join(' ')
    return data


def create_ner_dict(text: List[str]) -> DefaultDict:
    nlp = spacy.load('en_core_web_md')
    ner_pipeline = nlp.pipe(text, disable=["tagger", "parser"])
    entity_dict = defaultdict(list)
    for doc in ner_pipeline:
        for entity in doc.ents:
            entity_dict[str(entity.label_)].append(str(entity.text))

    # Reduce to just unique values
    for key in entity_dict.keys():
        entity_dict[key] = sorted(list(set(entity_dict[key])))
    return entity_dict


def compute_doc_frequency_matrix(corpus: List[str]) -> pd.DataFrame:
    vec = CountVectorizer()
    docs = vec.fit_transform(corpus)
    words = vec.get_feature_names()

    # builds the matrix as a list of dicts
    matrix = []
    for arr in docs.toarray():
        row = dict(zip(words, arr))
        matrix.append(row)
    # Returns as a Pandas DataFrame
    return pd.DataFrame(matrix)


def compute_inverse_doc_matrix(corpus: List[str]) -> pd.DataFrame:
    vec = TfidfVectorizer()
    docs = vec.fit_transform(corpus)
    words = vec.get_feature_names()

    # builds the matrix as a list of dicts
    matrix = []
    for arr in docs.toarray():
        row = dict(zip(words, arr))
        matrix.append(row)
    # Returns as a Pandas DataFrame
    return pd.DataFrame(matrix)


def run_ner_parse(corpus: List[str]) -> List[str]:
    nlp = spacy.load('en_core_web_md')
    ner_pipeline = nlp.pipe(corpus, disable=["tagger", "parser"])
    ner_types = ['PERSON', 'NORP', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']  # NER types to filter

    # re-build the text corpus with just found entities within the NER types
    ner_corpus = []
    for doc in ner_pipeline:
        found_entities = []
        for entity in doc.ents:
            if entity.label_ in ner_types:
                found_entities.append(entity.text)
        if len(found_entities) > 0:
            ner_corpus.append(' '.join(found_entities))
    return ner_corpus


def run_prop_noun_parse(corpus: List[str]) -> List[str]:
    nlp = spacy.load('en_core_web_md')
    dep_pipeline = nlp.pipe(corpus, disable=['ner'])

    # re-build the text corpus with just Proper Nouns
    noun_corpus = []
    for doc in dep_pipeline:
        # Filter out STOP WORDS and all other POS tags
        found_pronouns = [token.text for token in doc if token.is_stop is False and token.pos_ == 'PROPN']
        if len(found_pronouns) > 0:
            noun_corpus.append(' '.join(found_pronouns))
    return noun_corpus


def run_ner_noun_parse(corpus: List[str]) -> List[str]:
    nlp = spacy.load('en_core_web_md')
    ner_pipeline = nlp.pipe(corpus)
    ner_types = ['PERSON', 'NORP', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']  # NER types to filter
    noun_types = ['PROPN', 'NOUN']

    # re-build the text corpus with just found entities within the NER types
    ner_corpus = []
    for doc in ner_pipeline:
        # Get all nouns and proper nouns
        noun_tokens = [token for token in doc if token.pos_ in noun_types]

        # Get entities from nouns
        entity_tokens = [token for token in noun_tokens if token.ent_type_ in ner_types]

        # Lemmatization of text (shortens it)
        lemma_tokens = [token.lemma_.strip().lower() for token in entity_tokens if token.is_stop is False
                        and token.text not in punctuation]

        ner_corpus.append(' '.join(lemma_tokens))
    return ner_corpus


def simple_nlp_tokenize(corpus):
    nlp = spacy.load('en_core_web_lg')
    pipeline = nlp.pipe(corpus)
    for idx, doc in enumerate(pipeline):
        lemma_tokens = [token.lemma_.strip().lower() for token in doc if token.is_stop is False
                        and token.text not in punctuation and token.lemma_ != '-PRON-']

        corpus[idx] = ' '.join(lemma_tokens)
    return corpus
