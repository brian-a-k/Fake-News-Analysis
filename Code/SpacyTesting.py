import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")

# Processed Document
doc = nlp(text)

# Analyze syntax
print('NOUNS:')
for chunk in doc.noun_chunks:
    print("Noun phrase:", chunk.text)


print('\nVERBS:')
for token in doc:
    if token.pos_ == 'VERB':
        print("Verbs:", token.lemma_)

# Find named entities, phrases and concepts
print('\nENTITIES:')
for entity in doc.ents:
    print('NER:', entity.text, ':', entity.label_)


