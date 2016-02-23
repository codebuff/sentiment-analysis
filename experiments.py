from sklearn.naive_bayes import MultinomialNB

from sa_multinomial_nb import SA_MultinomialNB
from sa_sgd_classifier import SA_SGDClassifier


def get_accuracy_vs_dataset_size_NB():
    percentage = 10
    while percentage != 100:
        nb = SA_MultinomialNB(train_data_percentage=percentage, starting_pos=00)
        nb.train()
        nb.predict_from_test_data()
        percentage += 10


def get_accuracy_vs_dataset_size_SGD():
    percentage = 10
    while percentage != 100:
        nb = SA_MultinomialNB(train_data_percentage=percentage, starting_pos=00, count_vector_type='other')
        nb.train()
        nb.predict_from_test_data()
        percentage += 10


def feature_aberration():
    nb = SA_MultinomialNB(count_vector_type='other', starting_pos=0)
    freq_mat = nb.counts
    nb.counts = freq_mat[:, 0]
    nb.classifier = MultinomialNB()
    targets = nb.training_data['class'].values
    counts_tf_idf = nb.tf_idf.fit_transform(nb.counts)
    nb.classifier.fit(counts_tf_idf, targets)
    nb.trained = True
    nb.predict_from_test_data()


get_accuracy_vs_dataset_size_NB()