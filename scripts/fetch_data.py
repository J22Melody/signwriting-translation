import re
import random
import math

import tensorflow_datasets as tfds
import sign_language_datasets.datasets

import nltk
nltk.download('punkt')


random.seed(42)

def parse(raw, includeSpatials=False):
    sequence = []

    # if not punctuation, add the box information
    if includeSpatials and not raw.startswith('S'):
        for ch2 in raw[0:8]:
            sequence.append(ch2)

    for index, ch in enumerate(raw):
        if ch == 'S':
            # sequence of symbols
            sequence.append(raw[index:index + 6])

            # spatials
            if includeSpatials:
                for ch2 in raw[index + 6:index + 13]:
                    sequence.append(ch2)

    return ' '.join(sequence)

signbank = tfds.load(name='sign_bank')['train']
data_list = []
count = {'total': 0}

# see https://github.com/sign-language-processing/datasets/blob/master/sign_language_datasets/datasets/signbank/signbank.py#L164
for index, row in enumerate(signbank):
    puddle_id = row['puddle'].numpy().item()
    assumed_spoken_language_code = row['assumed_spoken_language_code'].numpy().decode('utf-8')
    country_code = row['country_code'].numpy().decode('utf-8')

    # ASL to English
    if country_code == 'us' and assumed_spoken_language_code == 'en':
        terms = [f.decode('utf-8') for f in row['terms'].numpy()]

        # print('#', puddle_id)
        # print(terms)
        # continue

        en = ''
        isDict = False

        # Sentences: Literature US, ASL Bible Books NLT, ASL Bible Books Shores Deaf Church
        if puddle_id == 5 or puddle_id == 151 or puddle_id == 152:
            # the first element is the title
            # the second is the main text
            if len(terms) > 1:
                # only take the main text
                en = terms[1]

                if puddle_id == 151:
                    # remove line-break and source, e.g., Nehemiah 3v11 NLT
                    en = re.sub(r"\n\n.*NLT", "", en)

        # Dictionaries: Dictionary US, LLCN & SignTyp, ASL Bible Dictionary
        if puddle_id == 4 or puddle_id == 25 or puddle_id == 28:
            # longest word allowed
            if len(terms) > 0 and len(terms[0]) < 100:
                en = terms[0]

            isDict = True

        if not en or en.startswith('<iframe'):
            continue

        # tokenize
        en_tokenized = ' '.join(nltk.word_tokenize(en))

        sign_sentence = row['sign_writing'].numpy().decode('utf-8')

        if not sign_sentence:
            continue

        # run customized parser
        signs = sign_sentence.split(' ')
        sign = ' '.join(list(map(parse, signs)))
        # sign_plus = ' '.join(list(map(lambda x: parse(x, True), signs)))

        data_list.append({
            'isDict': isDict, 
            'en': en.encode("unicode_escape").decode("utf-8"),
            'en_tokenized': en_tokenized.encode("unicode_escape").decode("utf-8"),
            'sign': sign,
            # 'sign+': sign_plus,
        })

        count['total'] += 1
        count[puddle_id] = count[puddle_id] + 1 if puddle_id in count else 1
        print(count['total'])

from pprint import pprint
pprint(count)

random.shuffle(data_list)

total_size = len(data_list)
train_size = math.floor(total_size*0.95)
dev_size = math.floor(total_size*0.03)
test_size = total_size - train_size - dev_size

data_list = sorted(data_list, key=lambda d: d['isDict']) 

dev = data_list[:dev_size]
test = data_list[dev_size:test_size + dev_size]
train = data_list[test_size + dev_size:]

random.shuffle(train)

with open('./data_full/train.sign', 'w+') as f_sign:
    with open('./data_full/train.sign+', 'w+') as f_sign_plus:
        with open('./data_full/train.withDict.sign', 'w+') as f_sign_withDict:
            with open('./data_full/train.en', 'w+') as f_en:
                with open('./data_full/train.tokenized.en', 'w+') as f_en_tokenized:
                    with open('./data_full/train.withDict.en', 'w+') as f_en_withDict:
                        with open('./data_full/train.withDict.tokenized.en', 'w+') as f_en_withDict_tokenized:
                            for item in train:
                                if not item['isDict']:
                                    f_en.write("%s\n" % item['en'])
                                    f_en_tokenized.write("%s\n" % item['en_tokenized'])
                                    f_sign.write("%s\n" % item['sign'])
                                    # f_sign_plus.write("%s\n" % item['sign+'])
                                f_en_withDict.write("%s\n" % item['en'])
                                f_en_withDict_tokenized.write("%s\n" % item['en_tokenized'])
                                f_sign_withDict.write("%s\n" % item['sign'])
                    

with open('./data_full/dev.sign', 'w+') as f_sign:
    with open('./data_full/dev.sign+', 'w+') as f_sign_plus:
        with open('./data_full/dev.en', 'w+') as f_en:
            with open('./data_full/dev.tokenized.en', 'w+') as f_en_tokenized:
                for item in dev:
                    f_sign.write("%s\n" % item['sign'])
                    # f_sign_plus.write("%s\n" % item['sign+'])
                    f_en.write("%s\n" % item['en'])
                    f_en_tokenized.write("%s\n" % item['en_tokenized'])

with open('./data_full/test.sign', 'w+') as f_sign:
    with open('./data_full/test.sign+', 'w+') as f_sign_plus:
        with open('./data_full/test.en', 'w+') as f_en:
            with open('./data_full/test.tokenized.en', 'w+') as f_en_tokenized:
                for item in test:
                    f_sign.write("%s\n" % item['sign'])
                    # f_sign_plus.write("%s\n" % item['sign+'])
                    f_en.write("%s\n" % item['en'])
                    f_en_tokenized.write("%s\n" % item['en_tokenized'])