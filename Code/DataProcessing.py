from typing import *
import pandas as pd
from collections import defaultdict
import spacy


def clean_columns(raw_cols: List[str]) -> List[str]:
    return [name.lower().replace(' ', '_') for name in raw_cols]


def fill_null_string_columns(data: pd.DataFrame, fill_dict: dict):
    return data.fillna(value=fill_dict)


# Removes all characters that are NOT: (letters numbers . ? , !)
def clean_text_column(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    data[column_name] = data[column_name].str.replace('[^0-9a-zA-Z’ .!?,]', '', regex=True)
    return data


def create_ner_dict(text: List[str]) -> DefaultDict:
    nlp = spacy.load('en_core_web_md')
    ner_pipeline = nlp.pipe(text, disable=["tagger", "parser"])
    entity_dict = defaultdict(list)
    for doc in ner_pipeline:
        for entity in doc.ents:
            entity_dict[str(entity.label_)].append(str(entity.text))
    return entity_dict







