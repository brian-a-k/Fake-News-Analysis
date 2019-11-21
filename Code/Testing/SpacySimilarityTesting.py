import spacy
import numpy as np

nlp = spacy.load('en_core_web_lg')  # load large model

doc = nlp('Trump president white house')

doc_2 = nlp('I love my dog Lola')

# for token in doc:
#     print(token.text, token.has_vector, token.vector_norm, token.is_oov)


#print(doc.similarity(doc_2))

vec_full = np.mean(doc.vector)

vec_norm = doc.vector_norm
print(vec_norm)
print(vec_full)



