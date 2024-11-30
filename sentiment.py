"""
CS311 Project Song Lyrics Sentiment Analysis

Full Name: Frank Bautista

"""
import argparse, math, string
from typing import Generator, Hashable, Iterable, List, Sequence, Tuple
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split # for splitting the dataset into training and testing sets


class MultiClassNaiveBayes():
    def __init__(self, labels):
        self.labels = labels
        self.word_frequencies = {label: {} for label in self.labels}
        self.document_counts = {label: 0 for label in self.labels}
        self.total_document_count = 0
        self.total_words_for_label = {label: 0 for label in self.labels}

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess a text document by lowercasing, removing punctuation, and splitting into words

        Args:
            text (str): Input text

        Returns:
            List[str]: List of preprocessed words
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        text = text.strip() # remove leading and trailing whitespaces
        return text.split()


    
    def add_example(self, example: str, labels: List[Hashable]):
        """
        Add a single training example with label to the model

        Args:
            example (str): Text input
            label List[Hashable]: Example labels (multi-class)

        """
        stripped_example = self.preprocess(example)
        for label in labels: # if label is present, increment document count and word frequency
            self.document_counts[label] += 1
            self.total_document_count += 1
            for word in stripped_example:
                self.total_words_for_label[label] += 1
                if word in self.word_frequencies[label]:
                    self.word_frequencies[label][word] += 1
                else:
                    self.word_frequencies[label][word] = 1

    def predict(self, example: str, pseudo=0.0001, threshold=0.01) -> Sequence[float]:
        """
        Predict the P(label|example) for example text, return probabilities as a sequence

        Args:
            example (str): Test input
            pseudo (float, optional): Pseudo-count for Laplace smoothing. Defaults to 0.0001.

        Returns:
            Sequence[float]: Probabilities in order of originally provided labels
        """
        stripped_example = self.preprocess(example)

        # Calculate the prior probabilities and conditional probabilities for each label
        prior_probabilities = {label: math.log(self.document_counts[label] / sum(self.document_counts.values()) ) for label in self.labels}
        conditional_probabilities = {label: self.conditional_probability(stripped_example, label, pseudo) for label in self.labels}
        naive_bayes_denominator = np.logaddexp.reduce([prior_probabilities[label] + conditional_probabilities[label] for label in self.labels])

        # key = label, value = P(label|example)
        label_probabilities = {label: math.exp(prior_probabilities[label] + conditional_probabilities[label] - naive_bayes_denominator) for label in self.labels}
    
        # Applying the Threshold
        for label in self.labels:
            if label_probabilities[label] < threshold:
                label_probabilities[label] = 0

        # Only return labels with non-zero probabilities
        return [label for label in self.labels if label_probabilities[label] > 0]
    
    def conditional_probability(self, words, label, pseudo=0.0001) -> float:
        """
        Given a list of words, find the conditional probability of the features (words) given the sentiment using the Naive Bayes model.

        Args:
            words (list): list of preprocessed words. The features.
            sentiment (int): the sentiment to calculate the conditional probability for. 0 for negative, 1 for positive
            pseudo (float): Pseudo-count for Laplace smoothing. Defaults to 0.0001.
        
        Returns:
            float: the conditional probability of the features given the sentiment. (log probability to avoid underflow)

        """
        accumulation_of_probabilities = 0

        for word in words:
            word_count = self.word_frequencies[label].get(word, 0)
            # Using Laplace smoothing to avoid zero probabilities
            probability = (word_count + pseudo) / (self.total_words_for_label[label] + pseudo * len(self.word_frequencies[label]) )
            accumulation_of_probabilities += math.log(probability)

        return accumulation_of_probabilities



def compute_metrics(y_true, y_pred):
    """Compute metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.

    Returns:
        dict: Dictionary of metrics in including confusion matrix, accuracy, recall, precision and F1
    """
    return {
        "confusion": metrics.confusion_matrix(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
    }


if __name__ == "__main__":

    full_dataset = pd.concat([pd.read_csv("goEmotionData\goemotions_1.csv"), pd.read_csv("goEmotionData\goemotions_2.csv"), pd.read_csv("goEmotionData\goemotions_3.csv")]) 
    filtered_dataset =  full_dataset[full_dataset['example_very_unclear'] == 0] # remove unclear examples that provide no labels.
    train_data, test_data = train_test_split(filtered_dataset, test_size=0.2, random_state=42) # 42 is the answer to everything and also the seed here.

    parser = argparse.ArgumentParser(description="Train Naive Bayes sentiment analyzer")

    parser.add_argument("example", nargs="?", default=None)
    args = parser.parse_args()

    # Train model
    model = MultiClassNaiveBayes(labels=filtered_dataset.columns[9:].tolist())

    for _, row in train_data.iterrows():
        text  = row['text'] # extract text from the 'text' column
        emotion_labels = [label for label in train_data.columns[9:] if row[label] == 1] 
        model.add_example(text, emotion_labels)
    if args.example:
        print(model.predict(args.example))
    else:
        # Predict on test_data
        y_true = []
        y_pred = []
        for _, row in test_data.iterrows():
            text = row['text']
            emotion_labels = [label for label in test_data.columns[9:] if row[label] == 1]
            predicted_labels = model.predict(text)

            y_true.append(emotion_labels)
            y_pred.append(predicted_labels)

        # Binarize the true and predicted labels for evaluation
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=test_data.columns[9:])
        y_true_bin = mlb.fit_transform(y_true)
        y_pred_bin = mlb.transform(y_pred)

        # Print classification metrics
        from sklearn.metrics import classification_report
        print("Multi-Class Classification Report:")
        print(classification_report(y_true_bin, y_pred_bin, target_names=test_data.columns[9:]))


