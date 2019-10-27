from __future__ import unicode_literals, print_function

import spacy


def extract_currency_relations(doc):
    # Merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = spacy.util.filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    relations = []
    for money in filter(lambda w: w.ent_type_ == "MONEY", doc):
        if money.dep_ in ("attr", "dobj"):
            subject = [w for w in money.head.lefts if w.dep_ == "nsubj"]
            if subject:
                subject = subject[0]
                relations.append((subject, money))
        elif money.dep_ == "pobj" and money.head.dep_ == "prep":
            relations.append((money.head.head, money))
    return relations


TEXTS = [
    "Net income was $9.4 million compared to the prior year of $2.7 million.",
    "Revenue exceeded twelve billion dollars, with a loss of $1b."
]


def main(model="en_core_web_md"):
    nlp = spacy.load(model)
    ent_relation_pipeline = nlp.pipe(TEXTS)

    for doc in ent_relation_pipeline:
        relations = extract_currency_relations(doc)
        for r1, r2 in relations:
            print('TEXT PHRASE:', r1.text, 'ENT_TYPE:', r2.ent_type_, 'ENT_TEXT:', r2.text)


if __name__ == '__main__':
    # todo modify this for other NER relations!!
    main()
