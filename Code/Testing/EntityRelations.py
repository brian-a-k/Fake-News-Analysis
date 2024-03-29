import spacy


fake_headlines = [
    'The Amish In America Commit Their Vote To Donald Trump Mathematically Guaranteeing Him A Presidential Victory  ABC News',
    'Obama Signs Executive Order Declaring Investigation Into Election Results Revote Planned For Dec. 19th  ABC News',
    'Comment on HALLOWEEN IN THE CASTRO RETURNS IN 2014! by Day of the Dead 2015 History, food and reflections  Andrea Lawson Gray',
    'Comment on Tutorial Riding The Philippine Jeepney by Ivan Jose',
    'Comment on What White House Executive Chef Comerford Would Cook For President Hillary’s First Meal by Tony Rabon',
    'Comment on Philippines Voids Building Permit Of Trump Tower In Makati City by Shirley Barnett',
    'Comment on Hillary Clinton Campaign Logo Has A Subliminal Message by    ',
    'Comment on Philippine Government To Take Back The Internet’ From Maria Ressa And Rappler? by adobochron',
    'Comment on If Elected President, Donald Trump Will Not Live In The White House by Rebecca Bennett',
    'Comment on WHITE HOUSE EXECUTIVE CHEF REVEALS OBAMA’S FAVORITE FILIPINO FOODS by Eligio Abellera',
    'Comment on Respected Journalist Bill Moyers Is First HighProfile American To Flee U.S. After Trump’s Election by Sue Penn SuePennonTwitte',
    'Comment on Donald Trump To Replace Filipina White House Chef With Paula Deen by Mitch Jalandoni',
    'Apple Adds AltRight’ Key To Its Computer Keyboards ',
    'Comment on About Us by adobochronicles.com  Real or Satire?',
    'Comment on Philippines’ Department Of Tourism Unveils New Promo Slogan, A Double Entendre by KoelWritingSouls',
    'Wikileaks Gives Hillary An Ultimatum QUIT, Or We Dump Something LifeDestroying  The Resistance The Last Line of Defense',
    'Hillary Collapses On Her Way To The Stage, Sellout Bruce Springsteen Covers For Her  The Resistance The Last Line of Defense',
    'Van Full Of Illegals Shows Up To Vote Clinton At SIX Polling Places, Still Think Voter Fraud Is A Myth?  The Resistance The Last Line of Defense',
    'Lady Gaga’s Twitter Attack On Melania Trump Lands Her In Handcuffs When The Two Meet Face To Face  The Resistance The Last Line of Defense']


def extract_proper_noun_relations(doc):
    # Merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = spacy.util.filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    ner_types = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'PRODUCT']
    relations = []
    for prop_noun in filter(lambda w: w.ent_type_ in ner_types, doc):
        if prop_noun.dep_ in ("attr", "dobj"):
            subject = [w for w in prop_noun.head.lefts if w.dep_ == "nsubj"]
            if subject:
                subject = subject[0]
                relations.append((subject, prop_noun))
        elif prop_noun.dep_ == "pobj" and prop_noun.head.dep_ == "prep":
            relations.append((prop_noun.head.head, prop_noun))
    return relations


def main(model="en_core_web_md"):
    nlp = spacy.load(model)
    ent_relation_pipeline = nlp.pipe(fake_headlines)

    for i, doc in enumerate(ent_relation_pipeline):
        relations = extract_proper_noun_relations(doc)
        for r1, r2 in relations:
            print('ID:', i, 'TEXT PHRASE:', r1.text, 'ENT_TYPE:', r2.ent_type_, 'ENT_TEXT:', r2.text)


if __name__ == '__main__':
    # todo modify this for other NER relations!!
    main()
