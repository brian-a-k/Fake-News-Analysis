import spacy
import pandas as pd
import numpy as np


# Removes all characters that are NOT: (letters numbers . ? , ! ')
def clean_text_column(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    data[column_name] = data[column_name].str.replace('[^0-9a-zA-Z .!?,]', '', regex=True)
    return data


def get_fake_data():
    fake_news = pd.read_csv('/Users/briankalinowski/Desktop/Data/Kaggle/real_news.csv')
    fake_news.drop(columns=['uuid', 'ord_in_thread', 'published'], inplace=True)
    fake_news = fake_news.dropna(subset=['text'])
    fake_news_content = fake_news[['title', 'text', 'type']]
    fake_news_content = fake_news_content.reset_index(drop=True)
    fake_news_content.title = fake_news_content.title.apply(lambda row: str(row).lower())
    fake_news_content.text = fake_news_content.text.apply(lambda row: str(row).lower())
    fake_news_content = clean_text_column(fake_news_content, 'title')
    fake_news_content = clean_text_column(fake_news_content, 'text')
    return list(fake_news_content[fake_news_content.type == 'fake']['text'])


# Load English tokenizer, tagger, parser, NER and token vectors
nlp = spacy.load("en_core_web_lg")

text = get_fake_data()

doc = nlp(text[2])

# Find named entities, phrases and concepts
ner_dict = {}

print('\nENTITIES:')
for entity in doc.ents:
    print('NER:', entity.text, ':', entity.label_)





