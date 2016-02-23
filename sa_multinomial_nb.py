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
        self.train_data_percentage = train_data_percentage
        self.data_base_path = data_base_path
        self.starting_pos = starting_pos
        self.total_entries = total_entries
        self.counts = None
        self.classifier = None
        if count_vector_type == 'normal':
            self.count_vectorizer = CountVectorizer()
        else:
            # just put anything different than 'normal' count_vector_type while initializing
            self.count_vectorizer = CountVectorizer(ngram_range=(1, 2))
        self.tf_idf = TfidfTransformer()
        self.initialized = False
        self.trained = False
        self.initialize()

    def initialize(self):
        max_training_datapoints = int(self.total_entries * (self.train_data_percentage / 100))
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
        base_path = './txt_sentoken/'
        test_reviews = []
        true_outputs = []
        for polarity in self.test_data_files:
            for file_name in self.test_data_files[polarity]:
                file = open(base_path + polarity + '/' + file_name, 'r')
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
            print('Enter the string:\n')
            string = input()

        string_list = [string]
        string_count = self.count_vectorizer.transform(string_list)
        predicted_polarity = self.classifier.predict(string_count)
        if true_output is not None:
            if true_output == predicted_polarity:
                print('correct prediction :)')
            else:
                print('incorrect prediction :(')
        print('predicted polarity:', predicted_polarity)
        #print('probability estimate', self.classifier.predict_proba(string_count))
        #print('log probability estimate', self.classifier.predict_log_proba(string_count))




test = SA_MultinomialNB(train_data_percentage=100, starting_pos=00, count_vector_type='other')
#test = SA_MultinomialNB(train_data_percentage=50, starting_pos=00)
test.train()
#test.predict_from_test_data()
# positive review
string = "one of my colleagues was surprised when i told her i was willing to see betsy's wedding . and she was shocked to hear that i actually liked it . her reaction was understandable when you consider that the film revolves around molly ringwald , who hasn't made a worthwhile film since 1986 . but the fact is , betsy's wedding is also an alan alda film . and while ringwald has been making duds for the last four years , alda has been involved with several noteworthy projects , including crimes and misdemeanors and a new life . written and directed by alda , betsy's wedding is a vibrant slice-of-life , mixing a few dramatic moments into a big bowl of whimsical humor . alda's comic elixir is smooth and refreshing--and a welcome change of pace from the usual summer fare . as bride and groom , molly ringwald and dylan walsh are the pivotal characters in the film , but they are by far the least interesting . walsh is a nonentity , with all the screen presence of a door knob . ringwald is simply unbearable and is easily the weakest link in the chain . she looks hideous with her short-cropped orange hair , red lip-stick and grotesque outfits . she's supposed to be a dress designer , but she looks more like a clown . and to make matters worse , ringwald's performance matches her appearance . thankfully , alda keeps ringwald's screen time to a minimum ; he is far more interested in the colorful periphery characters . the wedding is just a device to bring together the bride's working-class , italian family and the groom's rich , gentile family . ringwald's folks are homey and down-to-earth , with alda as her free-spirited father , madeline kahn as her practical mother , and ally sheedy as her lonely sister . walsh's clan , on the other hand , is prim , proper and ostentatious . when the two families meet and mingle , the movie becomes a story of culture clash , or as one character puts it , \" money versus values . \" ally sheedy , in a wonderfully understated performance , is one of the film's most pleasant surprises . sheedy expresses more with just her eyes than ringwald does with her entire body . it's anthony lapaglia , however , who seizes the spotlight . lapaglia plays stevie dee , a suave , overly polite mafioso who is formally courting sheedy with old-fashioned chivalry . lapaglia's sincere but dim-witted character is a riot . and what's uncanny is that lapaglia is a dead ringer for robert de niro , with a little bit of alec baldwin thrown in for good measure . lapaglia seems to have attended the de niro school of gangster acting , and his inspired performance is partly a tribute to his role-model and partly a rip-off . i don't know whether to say a star is born or a star is re-born , but i do know that lapaglia's over-the-top performance should not be missed . the scrumptious comic acting , however , extends well beyond sheedy and lapaglia . joe pesci , in particular , sinks his teeth into his role as alda's unscrupulous brother-in-law , a slum lord with mob ties , who is cheating on his wife ( catherine o'hara ) . alda , faced with challenge of both directing and acting , somehow finds just the right comic touch as the bride's financially-strapped father , a carpenter whose dreams are bigger than his wallet . the film adopts alda's psychological point of view as he tries to one : plan the wedding , and two : pay for it . as a filmmaker , alda's style of humor is remarkably restrained and tasteful . and while he doesn't have the comic genius of a woody allen , alda does possess the inspiration to make movies which are ten times more entertaining than the slop which usually passes for comedy ."

string = "With terrific craftsmanship, pure storytelling gusto and that Midas-touch ability to find grounds for optimism everywhere, Steven Spielberg has dramatised a true-life cold war spy-swap drama, starring Tom Hanks and Mark Rylance. Those brought up on John Le Carré might perhaps expect from this moral equivalence, shabby compromise and exhausted futility. But Spielberg, with his gift for uncynicism, uncovers decency and moral courage amidst all the Realpolitik. He works from an excellent screenplay by up-and-coming British dramatist Matt Charman, a script punched up in recognisable places by Joel and Ethan Coen." \
         "In 1962, America prepared to recover Gary Powers, the U2 spy-plane pilot captured by the Soviets. The plan was hand over their own incarcerated Russian spy Rudolf Abel, in a classic cold war prisoner exchange at dawn on the Glienecke bridge spanning East and West Berlin – the so-called “Bridge of Spies” – with snipers waiting on both sides ready to take their man out in case of last-second betrayal. Spielberg’s movie shows that the build-up involved agonisingly tense, deniable negotiations in bad faith, with each side calculating and re-calculating, with every day that went past, how likely it was that their man had cracked under interrogation, given up secrets, and therefore become valueless as an asset. To their opponents’ rage the US was actually insisting on a two-for-one: they also wanted an American student named Frederic Pryor wrongfully imprisoned in East Berlin, a deal which would make them look they’d had the best of the bargain." \
         "Austin Stowell plays Gary Powers, Mark Rylance plays the bespectacled and reserved Russian Abel, and Tom Hanks plays the civilian lawyer James Donovan who brokered the whole arrangement almost singlehandedly, and with pure amateur impulsiveness and stubbornness threw Pryor into the mix at the last moment. Donovan had accepted the poisoned chalice of being Abel’s state-sponsored public defender in the first place and persuaded the authorities that in Abel, America had the currency to buy back Powers. Sign up to our Film Today email Read more Hanks gives a very satisfying, watchable and assured performance, with just the right amount of hokum, homely and wile in judiciously balanced proportions. Jimmy Stewart gave us Mr Smith Goes to Washington; Hanks gives us Mr Donovan Goes to Cold War Berlin. Where his Donovan is bluffly ingenuous and straightforward, Rylance’s Russian spy Abel is a quietly voiced enigma, greeting the arresting officers in his chaotic Brooklyn apartment dressed in his underwear, asking meekly if he can put in his false teeth. (Details like these lead me to suspect the Coen brothers’ writing hand, and surely it had to be the Coens who created Abel’s bizarre fake “family members” in East Berlin, the phoney nearest-and-dearest the Communists have to produce as a cover for bringing Donovan to the Iron Curtain to start negotiating.) Repeatedly, Donovan will ask Abel in his prison cell: “Aren’t you worried?” and Abel will deadpan: “Would it help?” Rylance’s gentle, musical voice is a gift for this elegant, repeated gag. Hanks’s Donovan is a straight-arrow American guy, who has to square the circle of believing it to be his constitutional duty to defend a man who is in fact guilty of spying and attempting to undermine the American state – and its constitution. But Hanks shows how almost by accident, Donovan realises that Abel could be Uncle Sam’s ace in the hole. Once in East Berlin, Donovan has to negotiate the turf-war minefield of Soviet Russia and communist East Berlin, each of which has its own procedure and diplomatic amour propre. He amusingly has to deal with two deeply prickly Iron Curtain personalities: Pryor’s thin-skinned and mercurial lawyer Vogel (Sebastian Koch) and top DDR apparatchik (Burghart Klaussner), each of whom is given to office-based temper tantrums. Again: I wonder if it was the Coens who created these." \
         "The movie also gives us intriguing visual rhymes, duplications perhaps inspired by the Berlin wall. Spielberg withholds the facts about why Powers failed to commit suicide on capture with a poison-pin in a phoney coin – but instead shows us Abel removing secret information from a phoney coin of his own: somehow a dismal, petty skill. Donovan witnesses a horrible event from the window of his Berlin train, and finally sees something weirdly similar from his commuter train into Manhattan. And the subtle, underplayed, expertly directed “chase” sequence on the subway train at the beginning is terrifically good. Bridge of Spies has a brassy and justified confidence in its own narrative flair."
test.predict_from_user_string(string=string, true_output='pos')
#test.predict_from_user_string()
