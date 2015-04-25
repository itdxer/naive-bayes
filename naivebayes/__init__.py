import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class NaiveBayesTextClassifier(object):
    def __init__(self, categories, **kwargs):
        self.categories = categories
        self.vectorizer = CountVectorizer(**kwargs)

        # Will populate this variables in `train` method
        self.bag = None
        self.min_category_prob = None

    def train(self, documents, classes):
        total_docs = len(documents)
        categories = self.categories
        total_categories = len(categories)
        classes = np.array(list(map(categories.index, classes)))

        data = self.vectorizer.fit_transform(documents).toarray()

        row_combination_matrix = np.zeros((total_categories, total_docs))
        for i, category in enumerate(categories):
            row_combination_matrix[i, (classes == i)] = 1

        # Combine all words from one class
        data = np.dot(row_combination_matrix, data)
        number_of_words = data.shape[1]

        # Compute logarithmic probabilities
        words_in_categories = np.reshape((data != 0).sum(axis=1),
                                         (total_categories, 1))
        data = np.log((data + 1) / (words_in_categories + number_of_words))
        min_category_prob = np.log(
            1 / (words_in_categories + number_of_words)
        )

        self.bag = data
        self.min_category_prob = min_category_prob

    def classify(self, documents):
        if self.bag is None:
            raise AttributeError(
                "Your bag is empty. Train it before classify."
            )

        total_docs = len(documents)
        categories = self.categories
        vectorizer = self.vectorizer
        analyze = vectorizer.build_analyzer()

        data = vectorizer.transform(documents).toarray()
        counted_words_number = np.reshape(data.sum(axis=1), (total_docs, 1))
        probabilities = np.dot(self.bag, data.T)

        # `scikit-learn` ignore all words which we didn't use in train
        # examples. for this reason we must compute count of words again and
        # store them, we will balance probabilities with this information.
        total_words_number = np.zeros((total_docs, 1))
        for i, doc in enumerate(documents):
            total_words_number[i, :] = len(analyze(doc))

        ignored_words_number = total_words_number - counted_words_number
        probabilities += (ignored_words_number.T * self.min_category_prob)

        return list(map(categories.__getitem__, probabilities.argmax(axis=0)))
