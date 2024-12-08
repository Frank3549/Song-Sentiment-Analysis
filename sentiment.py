"""
CS311 Project Song Lyrics Sentiment Analysis (Multi-Label Naive Bayes)

Full Name: Frank Bautista

"""
import argparse, math
from typing import Generator, Hashable, Iterable, List, Sequence, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer 
from preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
USER_THRESHOLD = 0.05
USER_THRESHOLD_RANDOM = 0.50

class MultiLabelNaiveBayes():
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
            label List[Hashable]: Example labels

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

    def predict(self, example: str, pseudo=0.0001, threshold=0.50) -> Sequence[float]:
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
    full_dataset = pd.concat([pd.read_csv("goEmotionData/goemotions_1.csv"),
                               pd.read_csv("goEmotionData/goemotions_2.csv"),
                                pd.read_csv("goEmotionData/goemotions_3.csv")
                            ]) 
    full_dataset = full_dataset[full_dataset['example_very_unclear'] == 0] # remove unclear examples that provide no labels.

    # Extract the emotion labels
    emotion_labels = full_dataset.columns[9:].tolist()

    # Load the test dataset with new column names, process emotion_ids, and create a new column with the corresponding emotion labels
    test_data = pd.read_csv("goEmotionData/test.tsv", sep='\t', header=None, names=['text', 'emotion_ids', 'comment_id'])
    test_data['emotion_ids'] = test_data['emotion_ids'].apply(lambda x: x.split(','))
    test_data['emotion_labels'] = test_data['emotion_ids'].apply(lambda ids: [emotion_labels[int(id)] for id in ids])

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Naive Bayes Multi-Label sentiment analyzer")
    parser.add_argument("--example", nargs="?", default=None)
    parser.add_argument("--threshold_tuning", action="store_true", help="Automatically tune the threshold to find the best value to achieve the best accuracy") 
    parser.add_argument("--user", action="store_true", help="Run in interactive mode for real-time predictions")

    args = parser.parse_args()

    # Train model
    print("Training the model... This may take a while.")
    model = MultiLabelNaiveBayes(labels=emotion_labels)
    for _, row in full_dataset.iterrows():
        text  = row['text'] # extract text from the 'text' column
        labels_present = [label for label in emotion_labels if row[label] == 1] 
        model.add_example(text, labels_present)

    # Run interactive loop for user input
    if args.user:
        print("The model is ready. Type 'exit' to quit once done or 'r' to test a random example from the test dataset.\n")
        while True:
            user_input = input("Enter a sentence or text for emotion prediction (or 'r' for a random example): ")
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "r":
                # Select a random example from the test_data
                random_row = test_data.sample(1).iloc[0]
                random_text = random_row['text']
                random_labels = random_row['emotion_labels'] # true labels

                print(f"\nRandom Example: {random_text}")
                print(f"True Labels: {', '.join(random_labels)}")

                predicted_labels, original_probs = model.predict(random_text) # higher threshold for random examples

                # Sorting emotions by predicted probability
                sorted_emotions = sorted(original_probs.items(), key=lambda x: x[1], reverse=True)

                print("\nPredicted Emotions (sorted by confidence):")
                for label, prob in sorted_emotions:
                    if prob > USER_THRESHOLD:
                        print(f"{label} - {prob:.5f}")
                # Print all emotions sorted by probability for additional detail 
                print("\nAll Emotions (sorted by confidence):")
                for label, prob in sorted_emotions:
                    print(f"{label} - {prob:.5f}")
            else:
                predicted_labels, original_probs = model.predict(user_input, threshold=USER_THRESHOLD)  # lowered threshold for more predictions

                # Sorting emotions by their predicted probability (descending order)
                sorted_emotions = sorted(original_probs.items(), key=lambda x: x[1], reverse=True)

                if not predicted_labels:
                    print("\n-------Emotions could not confidently be detected, however the probabilities are:-------\n")
                    for label, prob in sorted_emotions:
                        print(f"{label} - {prob:.5f}")
                else:
                    print("\nPredicted Emotions:")
                    for label in predicted_labels:
                        print(f"{label} - {original_probs[label]:.5f}")

                    # Print all emotions sorted by probability for additional detail 
                    print("\nAll Emotions (sorted by confidence):")
                    for label, prob in sorted_emotions:
                        print(f"{label} - {prob:.5f}")
            print("\n")

    # Find the best threshold for the model using the test data (test data is formatted differently)
    elif args.threshold_tuning:

        best_threshold = 0
        best_accuracy = 0
        threshold_range = [i * 0.05 for i in range(1, 13)]  # From 0.05 to 0.60 in steps of 0.05

        for threshold in threshold_range: 
            y_true = []
            y_predicted = []

            for _, row in test_data.iterrows():
                text = row['text']
                predicted_labels, _ = model.predict(text, threshold=threshold)
                y_true.append(row['emotion_labels'])
                y_predicted.append(predicted_labels)

            # Evaluate the model performance
            mlb = MultiLabelBinarizer(classes=emotion_labels)
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
            predicted_labels, original_probabilities = model.predict(text)
            y_true.append(row['emotion_labels'])
            y_predicted.append(predicted_labels)

        # Binarize the true and predicted labels 
        mdl = MultiLabelBinarizer(classes=emotion_labels) 
        
        y_true_bin = mdl.fit_transform(y_true)
        y_pred_bin = mdl.transform(y_predicted)

        # Print classification metrics
        accuracy = accuracy_score(y_true_bin, y_pred_bin)
        print(f"\n------------Accuracy: {accuracy}------------\n")
        print("\n------------Classification Report:------------\n")
        print(classification_report(y_true_bin, y_pred_bin, target_names=emotion_labels, zero_division=0))
