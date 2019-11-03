import spacy
from typing import *
from string import punctuation

fake_headlines = [
    'The    Amish In America Commit Their Vote   To Donald Trump Mathematically Guaranteeing Him A Presidential Victory     ABC News',
    'Obama Signs Executive Order Declaring Investigation Into Election Results Revote Planned For Dec. 19th  ABC News',
    'Comment on HALLOWEEN IN THE CASTRO RETURNS   IN 2014! by Day of the Dead 2015 History,   food and reflections      Andrea Lawson Gray']

test = [' '.join(s.split()).lower() for s in fake_headlines]

for x in test:
    print(x)



# def run_ner_noun_parse(corpus: List[str]) -> List[str]:
#     nlp = spacy.load('en_core_web_md')
#     ner_pipeline = nlp.pipe(corpus)
#     ner_types = ['PERSON', 'NORP', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']  # NER types to filter
#     noun_types = ['PROPN', 'NOUN']
#
#     # re-build the text corpus with just found entities within the NER types
#     ner_corpus = []
#     for doc in ner_pipeline:
#         # Get all nouns and proper nouns
#         noun_tokens = [token for token in doc if token.pos_ in noun_types]
#
#         # Get entities from nouns
#         entity_tokens = [token for token in noun_tokens if token.ent_type_ in ner_types]
#
#         # Lemmatization of text (shortens it)
#         lemma_tokens = [token.lemma_.strip().lower() for token in entity_tokens if token.is_stop is False
#                         and token.text not in punctuation]
#
#         ner_corpus.append(' '.join(lemma_tokens))
#     return ner_corpus
#
#
# test = run_ner_noun_parse(fake_headlines)
# for x in test:
#     print(x)
