# Sentiment Analysis Project by Frank Bautista


# Optional Arguments:
```
  python3 sentiment.py -h

  options:
    -h, --help          show this help
                        message and exit
    --example [EXAMPLE] Provide your own text/example 
    --threshold_tuning  Automatically tune
                        the threshold to
                        find the best value
                        to achieve the best
                        accuracy
    --user              Run in interactive
                        mode for real-time
                        predictions
```
  Note: Running the file without any flags runs classification metrics.


# Example Output

```
Enter a sentence or text for emotion prediction (or press r for a random example): r
Random Example: "I miss you so much. It's hard to carry on."
True Labels: ['sadness', 'love']
Predicted Emotions:
sadness - 0.73214
love - 0.52103
neutral - 0.14567

```

## Downloading the required Stopwords library
 
In the terminal run the following:

```
pip install nltk
python (this will start the interactive python shell)
import nltk
nltk.download('stopwords')
exit()
```