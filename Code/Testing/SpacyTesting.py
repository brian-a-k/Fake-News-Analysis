import spacy
from typing import *
from string import punctuation


def assign_valid_class(row):
    # sum of the REAL and Fake scores (they are not probabilities)
    score_sum = (row.REAL + row.FAKE)

    # divide each score by their sum for weighted probabilities
    weighted_real = row.REAL / score_sum
    weighted_fake = row.FAKE / score_sum

    if weighted_real > weighted_fake:
        valid_class = 1
    elif weighted_real < weighted_fake:
        valid_class = 0
    else:
        # default to a real article if both weighted scores are equal (a lot of bias/vague articles in the raw_data-set)
        valid_class = 1
    return valid_class
