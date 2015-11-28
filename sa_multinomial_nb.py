import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score


class SA_MultinomialNB:
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
        from sklearn.naive_bayes import MultinomialNB
        self.classifier = MultinomialNB()
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
                file = open(base_path+polarity+'/'+file_name, 'r')
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




test = SA_MultinomialNB(train_data_percentage=50, starting_pos=00, count_vector_type='other')
#test = SA_MultinomialNB(train_data_percentage=50, starting_pos=00)
test.train()
#test.predict_from_test_data()
# positive review
string = "one of my colleagues was surprised when i told her i was willing to see betsy's wedding . and she was shocked to hear that i actually liked it . her reaction was understandable when you consider that the film revolves around molly ringwald , who hasn't made a worthwhile film since 1986 . but the fact is , betsy's wedding is also an alan alda film . and while ringwald has been making duds for the last four years , alda has been involved with several noteworthy projects , including crimes and misdemeanors and a new life . written and directed by alda , betsy's wedding is a vibrant slice-of-life , mixing a few dramatic moments into a big bowl of whimsical humor . alda's comic elixir is smooth and refreshing--and a welcome change of pace from the usual summer fare . as bride and groom , molly ringwald and dylan walsh are the pivotal characters in the film , but they are by far the least interesting . walsh is a nonentity , with all the screen presence of a door knob . ringwald is simply unbearable and is easily the weakest link in the chain . she looks hideous with her short-cropped orange hair , red lip-stick and grotesque outfits . she's supposed to be a dress designer , but she looks more like a clown . and to make matters worse , ringwald's performance matches her appearance . thankfully , alda keeps ringwald's screen time to a minimum ; he is far more interested in the colorful periphery characters . the wedding is just a device to bring together the bride's working-class , italian family and the groom's rich , gentile family . ringwald's folks are homey and down-to-earth , with alda as her free-spirited father , madeline kahn as her practical mother , and ally sheedy as her lonely sister . walsh's clan , on the other hand , is prim , proper and ostentatious . when the two families meet and mingle , the movie becomes a story of culture clash , or as one character puts it , \" money versus values . \" ally sheedy , in a wonderfully understated performance , is one of the film's most pleasant surprises . sheedy expresses more with just her eyes than ringwald does with her entire body . it's anthony lapaglia , however , who seizes the spotlight . lapaglia plays stevie dee , a suave , overly polite mafioso who is formally courting sheedy with old-fashioned chivalry . lapaglia's sincere but dim-witted character is a riot . and what's uncanny is that lapaglia is a dead ringer for robert de niro , with a little bit of alec baldwin thrown in for good measure . lapaglia seems to have attended the de niro school of gangster acting , and his inspired performance is partly a tribute to his role-model and partly a rip-off . i don't know whether to say a star is born or a star is re-born , but i do know that lapaglia's over-the-top performance should not be missed . the scrumptious comic acting , however , extends well beyond sheedy and lapaglia . joe pesci , in particular , sinks his teeth into his role as alda's unscrupulous brother-in-law , a slum lord with mob ties , who is cheating on his wife ( catherine o'hara ) . alda , faced with challenge of both directing and acting , somehow finds just the right comic touch as the bride's financially-strapped father , a carpenter whose dreams are bigger than his wallet . the film adopts alda's psychological point of view as he tries to one : plan the wedding , and two : pay for it . as a filmmaker , alda's style of humor is remarkably restrained and tasteful . and while he doesn't have the comic genius of a woody allen , alda does possess the inspiration to make movies which are ten times more entertaining than the slop which usually passes for comedy ."
test.predict_from_user_string(string=string, true_output='pos')
