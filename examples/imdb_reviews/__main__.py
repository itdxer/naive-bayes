import os
import time
import ntpath
from optparse import OptionParser

import numpy as np
from skll.metrics import kappa
from nltk.corpus import stopwords
from pandas.io.parsers import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from naivebayes import NaiveBayesTextClassifier


# -------------- Init options --------------- #

parser = OptionParser()
parser.add_option("-t", "--test",
                  action="store_true", dest="test", default=False,
                  help=("Split labeled data 70/30 ration and test "
                        "classification"))
parser.add_option("-p", "--predict",
                  action="store_true", dest="predict", default=False,
                  help="Predict test data")

(options, args) = parser.parse_args()

if options.test and options.predict:
    raise EnvironmentError(
        "You can run with `--test` or `--predict` option, not both"
    )

# -------------- Check data --------------- #

BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(BASEDIR, "data")

TEST_DATA_FILE = os.path.join(DATADIR, 'testData.tsv')
LABELED_TRAIN_DATA_FILE = os.path.join(DATADIR, 'labeledTrainData.tsv')
PREDICTED_DATA_FILE = os.path.join(DATADIR, 'predictedData.csv')

if not os.path.exists(DATADIR):
    os.mkdir(DATADIR)
    raise EnvironmentError(
        "Download data from "
        "https://www.kaggle.com/c/word2vec-nlp-tutorial/data "
        "and put it in {}.".format(DATADIR)
    )

important_files = (TEST_DATA_FILE, LABELED_TRAIN_DATA_FILE)
for tsv_file in important_files:
    if not os.path.exists(tsv_file):
        raise EnvironmentError("File {} doesn't exist at {}.".format(
            ntpath.basename(tsv_file), DATADIR
        ))

print("> Read train data")
train_data = read_csv(LABELED_TRAIN_DATA_FILE, sep='\t')

print("> Init classifier")
start_time = time.time()
classifier = NaiveBayesTextClassifier(
    categories=[0, 1],
    min_df=1,
    lowercase=True,
    # 127 English stop words
    stop_words=stopwords.words('english')
)

if options.test:
    print("> Split data to test and train")
    train_docs, test_docs, train_classes, test_classes = train_test_split(
        train_data.review, train_data.sentiment, train_size=0.7
    )

    print("> Train classifier")
    classifier.train(train_docs, train_classes)
    total_docs = len(train_docs)

elif options.predict:
    print("> Read test data")
    test_data = read_csv(TEST_DATA_FILE, sep='\t')

    print("> Train classifier")
    classifier.train(train_data.review, train_data.sentiment)
    total_docs = len(train_data)

print("-" * 42)
print("{:<25}: {:>6} articles".format("Total", total_docs))
print("{:<25}: {:>6} words".format(
    "Number of words", classifier.bag.shape[1]
))
print("{:<25}: {:>6.2f} seconds".format(
    "Parse time", time.time() - start_time
))
print("-" * 42)

# -------------- Classify --------------- #

print("> Start classify data")
start_time = time.time()

if options.test:
    predicted_classes = classifier.classify(test_docs)

    print(classification_report(test_classes, predicted_classes))
    print('-' * 42)
    print("{:<25}: {:>6} articles".format("Test data size", len(test_classes)))
    print("{:<25}: {:>6.2f} %".format(
        "Accuracy", 100 * accuracy_score(test_classes, predicted_classes))
    )
    print("{:<25}: {:>6.2f} %".format(
        "Kappa statistics", 100 * kappa(test_classes, predicted_classes)
    ))

elif options.predict:
    predicted_classes = classifier.classify(test_data.review)
    
    print("> Save predicted results")
    print("> {}".format(PREDICTED_DATA_FILE))
    np.savetxt(
        PREDICTED_DATA_FILE,
        np.concatenate(
            (test_data.values[:, 0:1], np.matrix(predicted_classes).T),
            axis=1
        ),
        delimiter=',', header='id,sentiment', comments='', fmt="%s"
    )
    print('-' * 42)


end_time = time.time()
print("{:<25}: {:>6.2f} seconds".format(
    "Computation time", end_time - start_time
))
print('-' * 42)