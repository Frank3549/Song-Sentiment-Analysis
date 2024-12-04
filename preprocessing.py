from typing import Generator, Hashable, Iterable, List, Sequence, Tuple
from nltk.corpus import stopwords
import string

NGRAM = 4

class TextPreprocessor:

    def __init__(self, stop_words_language: str = "english"):
        """
        Initialize the text preprocessor with optional stopword removal.

        Args:
            stop_words_language (str): Language for stopwords. Defaults to 'english'.
        """
        self.stop_words = set(stopwords.words(stop_words_language))

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess a text document by lowercasing, removing punctuation, and stopwords, and splitting into words.

        Args:
            text (str): Input text

        Returns:
            List[str]: List of preprocessed words
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        text = text.strip() # remove leading and trailing whitespaces
        text = text.split()
        filtered_text = [word for word in text if word not in self.stop_words] # remove stopwords
        filtered_text = self.unigram_bigram(filtered_text)

        return filtered_text

    def unigram_ngram (self, words: List[str], n: int ) -> List[str]:
        """
        Generate unigrams + n-grams.

        Args:
            words (List[str]): List of preprocessed words
            n (int): Size of n-grams
        
        Returns:
            List[str]: List of n-grams
        """

        n_grams = []
        for word in words:
            if len(word) > n:  
                n_grams.extend([word[i:i+n] for i in range(len(word) - n + 1)])
            else:
                n_grams.append(word) 
        return words + n_grams

    def ngram (self, words: List[str], n: int) -> List[str]:
        """
        Generate n-grams by splitting words into n-sized chunks or smaller if not long enough.

        Args:
            words (List[str]): List of preprocessed words
            n (int): Size of n-grams
        
        Returns:
            List[str]: List of n-grams
        """

        n_grams = []
        for word in words:
            if len(word) > n:  
                n_grams.extend([word[i:i+n] for i in range(len(word) - n + 1)])
            else:
                n_grams.append(word) 
        return n_grams
    
    def unigram_bigram_trigram (self, words: List[str]) -> List[str]:
        """
        Generate unigram + bigrams + trigram.

        Args:
            words (List[str]): List of preprocessed words
        
        Returns:
            List[str]: List of unigram + bigrams + trigram

        """
        bigrams = [' '.join([words[i], words[i+1]]) for i in range(len(words) - 1)]
        trigrams = [' '.join([words[i], words[i+1], words[i+2]]) for i in range(len(words) - 2)]
        return words + bigrams + trigrams
    
    def unigram_bigram (self, words: List[str]) -> List[str]: 
        """
        Generate bigrams by splitting words into 2 word chunks.

        Args:
            words (List[str]): List of preprocessed words
        
        Returns:
            List[str]: List of bigrams

        """
        bigrams = [' '.join([words[i], words[i+1]]) for i in range(len(words) - 1)]
        return words + bigrams

    def bigram (self, words: List[str]) -> List[str]: 
        """
        Generate bigrams by splitting words into 2 word chunks.

        Args:
            words (List[str]): List of preprocessed words
        
        Returns:
            List[str]: List of bigrams

        """
        bigrams = [' '.join([words[i], words[i+1]]) for i in range(len(words) - 1)]
        return bigrams