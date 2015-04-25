# Kaggle competition "Bag of Words Meets Bags of Popcorn"

Simple solution for [Kaggle competition](http://www.kaggle.com/c/word2vec-nlp-tutorial) using Naive Bayes calssifier. 

## Usage

Split train data in ratio 70/30 and check how algorithm train and test your sample result.

```bash
$ python imdb_reviews --test
```

Predict result for Kaggle competition tests anda save them in `data/predictedData.csv` file

```bash
$ python imdb_reviews --predict
```