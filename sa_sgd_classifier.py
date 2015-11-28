import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score


class SA_SGDClassifier:
    def __init__(self, train_data_percentage=70, count_vector_type='normal',
                 data_base_path=None, total_entries=1000, starting_pos=0):
        self.test_values = None
        self.training_data = None
        self.test_data_files = None
        self.train_data_percentage =train_data_percentage
        self.data_base_path = data_base_path
        self.starting_pos = starting_pos
        self.total_entries = total_entries
        self.counts = None
        self.classifier = None
        if count_vector_type == 'normal':
            self.count_vectorizer = CountVectorizer()
        else:
            # just put anything different than 'normal' count_vector_type while initializing
            self.count_vectorizer = CountVectorizer(ngram_range=(1,  2))
        self.tf_idf = TfidfTransformer()
        self.initialized = False
        self.trained = False
        self.initialize()

    def initialize(self):
        max_training_datapoints = int(self.total_entries * (self.train_data_percentage/100))
        if self.total_entries - max_training_datapoints < self.starting_pos:
            print('incorrect parameters provided check starting_pos and training_percentage')
            return
        temp = preprocessing.process_raw_data(max_training_datapoints=max_training_datapoints,
                                              starting_pos=self.starting_pos)
        self.training_data = temp[0]
        self.test_data_files = temp[1]
        self.counts = self.count_vectorizer.fit_transform(self.training_data['text'].values)
        self.initialized = True

    def train(self):
        if not self.initialized:
            print('classifier not initialized')
            return
        from sklearn.linear_model import SGDClassifier
        self.classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
        targets = self.training_data['class'].values
        counts_tf_idf = self.tf_idf.fit_transform(self.counts)
        self.classifier.fit(counts_tf_idf, targets)
        print('classifier was trained from', len(self.training_data['text']), 'entries')
        self.trained = True

    def predict_from_test_data(self):
        if not self.initialized or not self.trained:
            print('classifier not initialized or not trained')
            return
        base_path= './txt_sentoken/'
        test_reviews = []
        true_outputs = []
        for polarity in self.test_data_files:
            for file_name in self.test_data_files[polarity]:
                file = open(base_path+polarity+'/'+file_name,'r')
                file_content = file.read().splitlines()
                file.close()
                test_reviews.append(" ".join(file_content))
                true_outputs.append(polarity)

        test_reviews_count = self.count_vectorizer.transform(test_reviews)
        print('total reviews analyzed(test data):', len(2 * self.test_data_files['neg']))
        predictions = self.classifier.predict(test_reviews_count)
        print('f1 score for negative polarity:', f1_score(true_outputs, predictions, pos_label='neg'))
        print('f1 score for positive polarity:', f1_score(true_outputs, predictions, pos_label='pos'))
        print('mean accuracy on test data:', self.classifier.score(test_reviews_count, true_outputs))

    def predict_from_user_string(self, string=None, true_output=None):
        if not self.initialized or not self.trained:
            print('classifier not initialized or not trained')
            return
        if string is None:
            string = input(prompt='Enter the string:\n')

        string_list = [string]
        string_count = self.count_vectorizer.transform(string_list)
        predicted_polarity = self.classifier.predict(string_count)
        if true_output is not None:
            if true_output == predicted_polarity:
                print('correct prediction :)')
            else:
                print('incorrect prediction :(')
        print('predicted polarity:', predicted_polarity)
        print('probability estimate', self.classifier.predict_proba(string_count))
        print('log probability estimate', self.classifier.predict_log_proba(string_count))




#test = SA_SGDClassifier(train_data_percentage=90, starting_pos=00, count_vector_type='other')
test = SA_SGDClassifier(train_data_percentage=90, starting_pos=00)
test.train()
test.predict_from_test_data()
