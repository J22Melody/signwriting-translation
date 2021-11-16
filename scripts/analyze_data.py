import re
import random
import math

import tensorflow_datasets as tfds
import sign_language_datasets.datasets

signbank = tfds.load(name='sign_bank')['train']
data_list = []
count = {'total': 0}

# see https://github.com/sign-language-processing/datasets/blob/master/sign_language_datasets/datasets/signbank/signbank.py#L164
for index, row in enumerate(signbank):
    puddle_id = row['puddle'].numpy().item()
    assumed_spoken_language_code = row['assumed_spoken_language_code'].numpy().decode('utf-8')
    country_code = row['country_code'].numpy().decode('utf-8')
    pair_id = assumed_spoken_language_code + '-' + country_code
    terms = [f.decode('utf-8') for f in row['terms'].numpy()]
    sign_sentence = row['sign_writing'].numpy().decode('utf-8')

    count['total_raw'] = index

    if not assumed_spoken_language_code or not country_code or not sign_sentence or len(terms) < 1:
        continue

    count['total'] += 1

    if pair_id in count:
        count[pair_id]['total'] += 1

        if puddle_id in count[pair_id]:
            count[pair_id][puddle_id]['count'] += 1
        else:
            count[pair_id][puddle_id] = {
                'count': 1,
                # 'example': {
                #     'spoken': terms,
                #     'sign': sign_sentence,
                # },
            }
    else:
        count[pair_id] = {
            'total': 1,
            puddle_id: {
                'count': 1,
                # 'example': {
                #     'spoken': terms,
                #     'sign': sign_sentence,
                # },
            },
        }

from pprint import pprint

print('all country language pairs:')
pprint(count)

count_large = {}
print('threshold 1000:')
for key, value in count.items():
    if type(value) is int:
        continue
    if value['total'] > 1000:
        count_large[key] = value

pprint(count_large)

count_large = {}
print('threshold 10000:')
for key, value in count.items():
    if type(value) is int:
        continue
    if value['total'] > 10000:
        count_large[key] = value

pprint(count_large)
