# Naive Bayes Text Classifier

Text classifier based on Naive Bayes.

## Instalation

```bash
$ pip install naive-bayes
```

## Usage example

```python
from naivebayes import NaiveBayesTextClassifier

classifier = NaiveBayesTextClassifier(
    categories=categories_list,
    stop_words=stopwords_list
)
classifier.train(train_docs, train_classes)
predicted_classes = classifier.classify(test_docs)
```

`NaiveBayesTextClassifier` is a simple wrapper around `scikit-learn` class `CountVectorizer`. You can put all arguments which support this class. For more information please check `scikit-learn` official documentation.

## More examples

Check examples at `examples` folder. Before run them, install requirements in this folder.

Clone repository from github

```bash
$ git clone git@github.com:itdxer/naive-bayes.git
$ cd naive-bayes/examples
$ pip install -r requirements.txt
```

And run some example

### Usenet 20 newsgroup

```bash
$ python 20newsgroup
```

### Kaggle IMDB reviews competition

```bash
$ python imdb_reviews
```