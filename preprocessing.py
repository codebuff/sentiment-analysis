from os import listdir
from pandas import DataFrame

def get_data(base_path=None,max_datapoints=None):
    if max_datapoints is None:
        max_datapoints = -1
    if base_path is None:
        base_path='./txt_sentoken/'

    rows = []
    indexes = []

    polarities = ['neg', 'pos']

    for polarity in polarities:
        iterations_left = max_datapoints
        for file_name in listdir(base_path+polarity):
            file = open(base_path+polarity+'/'+file_name,'r')
            contents = file.read().splitlines()
            rows.append({'text': " ".join(contents), 'class': polarity})
            indexes.append(polarity+'_'+file_name)
            iterations_left -= 1
            if iterations_left == 0:
                break

    return DataFrame(rows, index=indexes)


def extract_features(data):
    from sklearn.feature_extraction.text import CountVectorizer
    count_vectorizer = CountVectorizer()
    return count_vectorizer.fit_transform(data['text'].values)

print(extract_features(get_data(max_datapoints=10)))
