"""
CS311 Project Song Lyrics Sentiment Analysis

Full Name: Frank Bautista

"""
import argparse, math
from typing import Generator, Hashable, Iterable, List, Sequence, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split # for splitting the dataset into training and testing sets
from sklearn.preprocessing import MultiLabelBinarizer 
from preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()

class MultiClassNaiveBayes():
    def __init__(self, labels):
        self.labels = labels
        self.word_frequencies = {label: {} for label in self.labels}
        self.document_counts = {label: 0 for label in self.labels}
        self.total_document_count = 0
        self.total_words_for_label = {label: 0 for label in self.labels}


    def add_example(self, example: str, labels: List[Hashable]):
        """
        Add a single training example with label to the model

        Args:
            example (str): Text input
            label List[Hashable]: Example labels (multi-class)

        """
        preprocessed_text = preprocessor.preprocess(example)

        for label in labels:
            self.document_counts[label] += 1
            self.total_document_count += 1
            for word in preprocessed_text:
                self.total_words_for_label[label] += 1
                if word in self.word_frequencies[label]:
                    self.word_frequencies[label][word] += 1
                else:
                    self.word_frequencies[label][word] = 1

    def predict(self, example: str, pseudo=0.0001, threshold=0.45) -> Sequence[float]:
        """
        Predict the P(label|example) for example text, return probabilities as a sequence

        Args:
            example (str): Test input
            pseudo (float, optional): Pseudo-count for Laplace smoothing. Defaults to 0.0001.

        Returns:
            (List[str], dict{str: float} ): List of labels over the threshold, and original dictionary of probabilities
        """
        preprocessed_text = preprocessor.preprocess(example)

        # Calculate the prior probabilities and conditional probabilities for each label
        prior_probabilities = {label: math.log(self.document_counts[label] / self.total_document_count ) for label in self.labels} #outputs are in log space
        conditional_probabilities = {label: self.conditional_probability(preprocessed_text, label, pseudo) for label in self.labels} #outputs are in log space
        total_odds = np.logaddexp.reduce([prior_probabilities[label] + conditional_probabilities[label] for label in self.labels])

        # key = label, value = P(label|example)
        original_label_probabilities = {label: math.exp(prior_probabilities[label] + conditional_probabilities[label] - total_odds) for label in self.labels}
        label_probabilities = original_label_probabilities.copy()

        # Applying the Threshold
        for label in self.labels:
            if label_probabilities[label] < threshold:
                label_probabilities[label] = 0

        # Only return labels with non-zero probabilities (those over the threshold) and the original probabilities dictionary
        return ([label for label in self.labels if label_probabilities[label] > 0], original_label_probabilities)
    
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
    
    # Load the dataset
    full_dataset = pd.concat([pd.read_csv("goEmotionData\goemotions_1.csv"), pd.read_csv("goEmotionData\goemotions_2.csv"), pd.read_csv("goEmotionData\goemotions_3.csv")]) 
    filtered_dataset =  full_dataset[full_dataset['example_very_unclear'] == 0] # remove unclear examples that provide no labels.
    
    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(filtered_dataset, test_size=0.1, random_state=42) # 42 is the answer to everything and also the seed here.

    parser = argparse.ArgumentParser(description="Train Naive Bayes Multi-class sentiment analyzer")
    parser.add_argument("example", nargs="?", default=None)
    parser.add_argument("--threshold_tuning", action="store_true", help="Automatically tune the threshold to find the best value to achieve the best accuracy") 
    args = parser.parse_args()

    # Train model
    model = MultiClassNaiveBayes(labels=filtered_dataset.columns[9:].tolist())
    for _, row in train_data.iterrows():
        text  = row['text'] # extract text from the 'text' column
        emotion_labels = [label for label in train_data.columns[9:] if row[label] == 1] 
        model.add_example(text, emotion_labels)
    
    # Predict on a single example
    if args.example:
        print(model.predict(args.example))

    # Find the best threshold for the model
    if args.threshold_tuning:

        best_threshold = 0
        best_accuracy = 0
        threshold_range = [i * 0.05 for i in range(1, 13)]  # From 0.05 to 0.60 in steps of 0.05

        for threshold in threshold_range: 
            y_true = []
            y_predicted = []

            for _, row in test_data.iterrows():
                text = row['text']
                emotion_labels = [label for label in test_data.columns[9:] if row[label] == 1]
                predicted_labels, _ = model.predict(text, threshold=threshold)
                y_true.append(emotion_labels)
                y_predicted.append(predicted_labels)

            # Evaluate the model performance
            mlb = MultiLabelBinarizer(classes=test_data.columns[9:])
            y_true_bin = mlb.fit_transform(y_true)
            y_pred_bin = mlb.transform(y_predicted)

            accuracy = accuracy_score(y_true_bin, y_pred_bin)

            # Update the best threshold
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold  

        print(f"Best threshold: {best_threshold} with accuracy: {best_accuracy:.4f}")

    # Predict on test_data and evaluate the model
    else:
        
        y_true = []
        y_predicted = []
        for _, row in test_data.iterrows():
            text = row['text']
            emotion_labels = [label for label in test_data.columns[9:] if row[label] == 1]
            predicted_labels, original_probabilities = model.predict(text)

            y_true.append(emotion_labels)
            y_predicted.append(predicted_labels)

        # Binarize the true and predicted labels 
        model = MultiLabelBinarizer(classes=test_data.columns[9:]) 
        
        y_true_bin = model.fit_transform(y_true)
        y_pred_bin = model.transform(y_predicted)

        #Print original probabilities and their labels:
        #print("output for one text example: %s" % text)
        #print(emotion_labels)
        for label in test_data.columns[9:]:
            print(label, round(original_probabilities[label], 8))
            #print(label, original_probabilities[label])
        
        # Print classification metrics
        print("Accuracy: ", accuracy_score(y_true_bin, y_pred_bin))
        print(classification_report(y_true_bin, y_pred_bin, target_names=test_data.columns[9:]))
