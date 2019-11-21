import spacy
import numpy as np
import sense2vec
from sense2vec import Sense2VecComponent


nlp = spacy.load('en_core_web_lg')

s2v = Sense2VecComponent()

nlp.add_pipe(s2v)

doc = nlp("A sentence about natural language processing.")


most_similar = doc[3:6]._.s2v_most_similar(3)

print(most_similar)

