from typing import *
import pandas as pd
import numpy as np
import spacy
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split


def split_training_testing_data(data: pd.DataFrame, target: str, size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Split raw_data with sklearn train/test
    x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=[target]),
                                                        data[target],
                                                        test_size=size,
                                                        random_state=21)

    # concat x, y raw_data-sets (for Spacy)
    data_training = pd.concat([x_train, y_train], 1).reset_index(drop=True)
    data_testing = pd.concat([x_test, y_test], 1).reset_index(drop=True)
    return data_training, data_testing


# Text Categorizer requires a iterable of tuples for training in the form of (text, score)
def format_training_data(df: pd.DataFrame, feature: str, target: str, limit: int, split=0.8) -> Tuple[Any, Any]:
    # Create training tuple list
    df['training_content'] = df.apply(lambda row: (row[feature], row[target]), axis=1)
    train_data_list = df['training_content'].tolist()

    # Shuffle list
    np.random.shuffle(train_data_list)
    train_data_list = train_data_list[-limit:]

    # Unpack into text and score labels
    texts, labels = zip(*train_data_list)

    # Text Cat. news labels
    text_cat_labels = [{"REAL": bool(y), "FAKE": not bool(y)} for y in labels]

    # Segment out raw_data for cross validation while training
    split = int(len(train_data_list) * split)
    train_tuple = (texts[:split], text_cat_labels[:split])
    eval_tuple = (texts[split:], text_cat_labels[split:])
    return train_tuple, eval_tuple


# Training Cross validation method
def evaluate_model_training(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1.0
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2.0 * (precision * recall) / (precision + recall)
    accuracy = ((tp + tn) / (tp + tn + fp + fn))
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score, 'textcat_a': accuracy}


# Make sure you have the spacy en_core_web_lg model downloaded
def get_base_model():
    nlp = spacy.load('en_core_web_lg')
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat', config={'exclusive_classes': True, 'architecture': 'simple_cnn'})
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")
    textcat.add_label("REAL")
    textcat.add_label("FAKE")
    return nlp


def main(train_data: list, eval_texts: Tuple, eval_cat: Tuple, n_iterations: int, save_path: str, verbose: bool = True):
    if save_path is None:
        print('ERROR no save dir for model')
        return

    nlp_training = get_base_model()
    # Only train the text categorizer
    other_pipes = [pipe for pipe in nlp_training.pipe_names if pipe != 'textcat']

    # main training loop
    with nlp_training.disable_pipes(*other_pipes):
        optimizer = nlp_training.begin_training()
        print('Training the model...')
        for i in range(n_iterations):
            loss_dict = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp_training.update(texts, annotations, sgd=optimizer, drop=0.2, losses=loss_dict)

            # Disable this if you do not want to see the CV scores every iteration (speeds this up)
            if verbose:
                textcat = nlp_training.get_pipe('textcat')
                with textcat.model.use_params(optimizer.averages):
                    scores = evaluate_model_training(nlp_training.tokenizer, textcat, eval_texts, eval_cat)
                    # output the CV scores for each iteration
                    print('Training Iteration: {}'.format(i + 1))
                    print('{:^5}\t{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F', 'A'))
                    print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3}\n'.format(loss_dict['textcat'],
                                                                                scores['textcat_p'],
                                                                                scores['textcat_r'],
                                                                                scores['textcat_f'],
                                                                                scores['textcat_a']))

    # Save the trained model
    if save_path is not None:
        nlp_training.to_disk(save_path)
        print('Training Complete!')
        print('Model Saved To:', save_path)


if __name__ == '__main__':
    # Edit for your local file (the tokenized fake/real news csv)
    load_path = '/Users/briankalinowski/Desktop/Data/sample_test.csv'

    # Edit for where you want to save the trained model
    save_model_path = '/Users/briankalinowski/Desktop/Data/spacy/test_model'

    # Tokenized DataFrame
    raw_data = pd.read_csv(load_path)

    # Train/Test split
    training_raw, testing_raw = split_training_testing_data(raw_data, 'valid_score', size=0.5)

    # Length (rows) of the raw_data
    n_texts = training_raw.shape[0]

    # Format the training raw_data
    (train_texts, train_cats), (dev_texts, dev_cats) = format_training_data(training_raw, 'tokenized_content',
                                                                            'valid_score', limit=n_texts)

    # Combine training text and labels
    train_final = list(zip(train_texts, [{'cats': cats} for cats in train_cats]))
    print('Loading news raw_data...')
    print('Using {} examples ({} for training, {} for cross evaluation)\n'.format(n_texts, len(train_texts),
                                                                                  len(dev_texts)))

    # Train the model
    main(train_final, dev_texts, dev_cats, n_iterations=5, save_path=save_model_path)
