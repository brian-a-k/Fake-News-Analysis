import pandas as pd
import spacy


class FakeNewsPredictions:
    def __init__(self, raw_test_data: pd.DataFrame, nlp_model_path: str):
        # Load Trained TextCat model saved to the path in TrainTextCategorizer.py
        self.trained_model = spacy.load(nlp_model_path, disable=['parser', 'tagger', 'ner'])
        self.test_data = raw_test_data

    def make_predictions(self, text_col: str, save_path: str, save_file: bool = True) -> pd.DataFrame:
        # list of each text article
        doc_list = [str(article) for article in self.test_data[text_col]]

        # trained pipeline
        text_cat_pipeline = self.trained_model.pipe(doc_list)

        # prediction scores as a DataFrame
        fake_vs_real_predictions = [doc.cats for doc in text_cat_pipeline]
        predictions_df = pd.DataFrame(fake_vs_real_predictions)

        # Apply class scoring metrics and final class (0, 1) prediction
        predictions_df['real_weighted_score'] = predictions_df.apply(real_weighted, axis=1)
        predictions_df['fake_weighted_score'] = predictions_df.apply(fake_weighted, axis=1)
        predictions_df['score_abs'] = predictions_df.apply(scores_abs, axis=1)
        predictions_df['valid_prediction'] = predictions_df.apply(assign_valid_class, axis=1)

        # final prediction DataFrame
        textcat_predictions = pd.concat([self.test_data, predictions_df], axis=1)

        # Save .csv file
        if save_file:
            textcat_predictions.to_csv(path_or_buf=save_path, header=True, index=None)
        return textcat_predictions


def scores_abs(row):
    return abs((row.REAL - row.FAKE))


def fake_weighted(row):
    return row.FAKE / (row.REAL + row.FAKE)


def real_weighted(row):
    return row.REAL / (row.REAL + row.FAKE)


def assign_valid_class(row):
    # sum of the REAL and Fake scores (they are not probabilities)
    score_sum = (row.REAL + row.FAKE)

    # divide each score by their sum for weighted probabilities
    weighted_real = row.REAL / score_sum
    weighted_fake = row.FAKE / score_sum

    if (weighted_real > weighted_fake) and (row.REAL >= 0.5) and (row.FAKE < 0.5):
        valid_class = 1
    elif (weighted_real < weighted_fake) and (row.FAKE >= 0.5) and (row.REAL < 0.5):
        valid_class = 0
    else:
        # just default to the raw scores
        if row.REAL > row.FAKE:
            valid_class = 1
        else:
            valid_class = 0
    return valid_class
