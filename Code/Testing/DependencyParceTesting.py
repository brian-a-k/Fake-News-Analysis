from typing import *
import pandas as pd
from collections import defaultdict
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


def run_prop_noun_parse(corpus: List[str]) -> List[str]:
    nlp = spacy.load('en_core_web_md')
    dep_pipeline = nlp.pipe(corpus, disable=['ner'])

    # re-build the text corpus with just Proper Nouns
    noun_corpus = []
    for doc in dep_pipeline:
        # Filter out STOP WORDS and all other POS tags
        found_pronouns = [token.text for token in doc if token.is_stop is False and token.pos_ == 'PROPN']
        if len(found_pronouns) > 0:
            noun_corpus.append(' '.join(found_pronouns))
    return noun_corpus


noun_test = run_prop_noun_parse(fake_headlines)
print(noun_test)
print(len(noun_test))
