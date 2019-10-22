import spacy

nlp = spacy.load('en_core_web_lg')  # load large model

tokens_1 = nlp('house car plane')

tokens_2 = nlp('door wheel wing')

for i in range(3):
    print(tokens_1[i].text, tokens_2[i].text, tokens_1[i].similarity(tokens_2[i]))



