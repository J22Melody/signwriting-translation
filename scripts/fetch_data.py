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

for index, row in enumerate(signbank):
    puddle_id = row['puddle'].numpy().item()

    # ASL Bible Books NLT and ASL Bible Books Shores Deaf Church
    # see https://github.com/sign-language-processing/datasets/blob/master/sign_language_datasets/datasets/signbank/signbank.py#L164
    if puddle_id == 151 or puddle_id == 152:
        terms = [f.decode('utf-8') for f in row['terms'].numpy()]

        # the first element is chapter
        # the second is the main text
        if len(terms) == 2:
            # only take the main text
            en = terms[1]
            # remove line-break and source, e.g., Nehemiah 3v11 NLT
            en = re.sub(r"\n\n.*NLT", "", en)
            # tokenize
            en = ' '.join(nltk.word_tokenize(en))

            sign = row['sign_writing'].numpy().decode('utf-8')

            # run standard js parser (https://github.com/sutton-signwriting/core/blob/master/src/fsw/fsw-parse.js#L63)
            # FIXME: js parser not compatible with dataset input
            # result = subprocess.run(['node', 'parse.js', sign], stdout=subprocess.PIPE)
            # sign = result.stdout

            # run customized parser
            signs = sign.split(' ')
            signs = list(map(parse, signs))
            sign = ' '.join(signs)

            data_list.append({
                'en': en.encode("unicode_escape").decode("utf-8"),
                'sign': sign,
                # 'sign+': sign,
            })

random.shuffle(data_list)

total_size = len(data_list)
train_size = math.floor(total_size*0.9)
dev_size = math.floor(total_size*0.09)

train = data_list[:train_size]
dev = data_list[train_size:train_size + dev_size]
test = data_list[train_size + dev_size:]

with open('./data/train.sign', 'w+') as f_sign:
    with open('./data/train.en', 'w+') as f_en:
        for item in train:
            f_sign.write("%s\n" % item['sign'])
            f_en.write("%s\n" % item['en'])

with open('./data/dev.sign', 'w+') as f_sign:
    with open('./data/dev.en', 'w+') as f_en:
        for item in dev:
            f_sign.write("%s\n" % item['sign'])
            f_en.write("%s\n" % item['en'])

with open('./data/test.sign', 'w+') as f_sign:
    with open('./data/test.en', 'w+') as f_en:
        for item in test:
            f_sign.write("%s\n" % item['sign'])
            f_en.write("%s\n" % item['en'])