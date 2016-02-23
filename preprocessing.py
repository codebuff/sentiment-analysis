from os import listdir
from pandas import DataFrame
import numpy


def process_raw_data(base_path=None, max_training_datapoints=None, starting_pos=0):
    if max_training_datapoints is None:
        max_training_datapoints = -1
    if base_path is None:
        base_path = './txt_sentoken/'

    rows = []
    indexes = []

    polarities = ['neg', 'pos']

    test_files = {polarities[0]: [], polarities[1]: []}
    for polarity in polarities:
        iterations_left = max_training_datapoints
        for file_name in listdir(base_path + polarity):
            if starting_pos > 0:
                starting_pos -= 1
                test_files[polarity].append(file_name)
                continue
            if iterations_left == 0:
                test_files[polarity].append(file_name)
                continue
            file = open(base_path + polarity + '/' + file_name, 'r')
            content = file.read().splitlines()
            file.close()
            rows.append({'text': " ".join(content), 'class': polarity})
            indexes.append(polarity + '_' + file_name)
            iterations_left -= 1

    data = DataFrame(rows, index=indexes)
    data = data.reindex(numpy.random.permutation(data.index))
    return data, test_files
