"""
CS311 Project Song Lyrics Sentiment Analysis

Full Name: Frank Bautista

"""
import argparse, math, string
from typing import Generator, Hashable, Iterable, List, Sequence, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split # for splitting the dataset into training and testing sets
from sklearn.preprocessing import MultiLabelBinarizer 
from nltk.corpus import stopwords

NGRAM = 5

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

    def n_gram (self, words: List[str], n: int):
        """Generate n-grams by splitting words into n-sized chunks or smaller if not long enough."""

        n_grams = []
        for word in words:
            if len(word) > n:  
                n_grams.extend([word[i:i+n] for i in range(len(word) - n + 1)])
            else:
                n_grams.append(word) 
        return n_grams
    
    def add_example(self, example: str, labels: List[Hashable]):
        """
        Add a single training example with label to the model

        Args:
            example (str): Text input
            label List[Hashable]: Example labels (multi-class)

        """
        stripped_example = self.preprocess(example)
        stripped_example = self.n_gram(stripped_example, NGRAM)
        for label in labels:
            self.document_counts[label] += 1
            self.total_document_count += 1
            for word in stripped_example:
                self.total_words_for_label[label] += 1
                if word in self.word_frequencies[label]:
                    self.word_frequencies[label][word] += 1
                else:
                    self.word_frequencies[label][word] = 1

    def predict(self, example: str, pseudo=0.0001, threshold=0.275) -> Sequence[float]:
        """
        Predict the P(label|example) for example text, return probabilities as a sequence

        Args:
            example (str): Test input
            pseudo (float, optional): Pseudo-count for Laplace smoothing. Defaults to 0.0001.

        Returns:
            Sequence[float]: Probabilities in order of originally provided labels
        """
        stripped_example = self.preprocess(example)
        stripped_example = self.n_gram(stripped_example, NGRAM)
        # Calculate the prior probabilities and conditional probabilities for each label
        prior_probabilities = {label: math.log(self.document_counts[label] / self.total_document_count ) for label in self.labels} #outputs are in log space
        conditional_probabilities = {label: self.conditional_probability(stripped_example, label, pseudo) for label in self.labels} #outputs are in log space
        total_odds = np.logaddexp.reduce([prior_probabilities[label] + conditional_probabilities[label] for label in self.labels])

        # key = label, value = P(label|example)
        label_probabilities = {label: math.exp(prior_probabilities[label] + conditional_probabilities[label] - total_odds) for label in self.labels}
    
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
            label (Hashable): the sentiment label.
            pseudo (float): Pseudo-count for Laplace smoothing. Defaults to 0.0001.
        
        Returns:
            float: the conditional probability of the features given the sentiment. (log probability to avoid underflow)

        """
        conditional_probability = 0

        for word in words:
            word_count = self.word_frequencies[label].get(word, 0)
            # Using Laplace smoothing to avoid zero probabilities
            probability = (word_count + pseudo) / (self.total_words_for_label[label] + pseudo * len(self.word_frequencies[label]) )
            conditional_probability += math.log(probability)

        return conditional_probability


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
        mlb = MultiLabelBinarizer(classes=test_data.columns[9:])
        y_true_bin = mlb.fit_transform(y_true)
        y_pred_bin = mlb.transform(y_pred)

        # Print classification metrics
        print("Accuracy: ", accuracy_score(y_true_bin, y_pred_bin))
        print(classification_report(y_true_bin, y_pred_bin, target_names=test_data.columns[9:]))
