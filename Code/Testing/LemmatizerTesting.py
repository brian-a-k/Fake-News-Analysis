import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS

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

punctuations = string.punctuation
stopwords = list(STOP_WORDS)
stopwords.append('Comment')

nlp = spacy.load('en_core_web_md')
nlp_pipeline = nlp.pipe(fake_headlines)
for i, doc in enumerate(nlp_pipeline):
    lemma_1 = [token for token in doc if token.lemma_ != "-PRON-"]
    lemma_2 = [token for token in lemma_1 if token.text not in stopwords and token.text.capitalize() not in stopwords
               and token.text not in punctuations]

    ner_types = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'PRODUCT']
    ner_s = [word for word in lemma_2 if word.ent_type_ in ner_types]
    print(lemma_2)
    print(list(doc.noun_chunks))
    print(ner_s)
    print('')

'''
# Parser for reviews
parser = English()
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_ for token in mytokens ]
    mytokens = [ token for token in mytokens if token not in stopwords and token not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens
'''
