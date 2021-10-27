import re
import random
import math

import tensorflow_datasets as tfds
import sign_language_datasets.datasets

import nltk
nltk.download('punkt')


random.seed(42)

def parse(raw):
    # raw: AS10011S10019S2e704S2e748M525x535S2e748483x510S10011501x466S2e704510x500S10019476x475
    # raw: M528x518S15a07472x487S1f010490x503S26507515x483 M509x515S18720491x486 S38800464x496 M522x557S18518458x542S30d00478x483S36d01480x517S18510488x542 S38700463x496 M531x529S15a56487x517S1fb20490x495S20600509x502S36d01479x472 M515x511S10047485x490 M537x522S1f750510x507S1f758474x507S2d201511x478S2d210464x478 M541x508S17620481x492S26512459x494S17628503x492S26506526x494 M562x537S2ff00482x483S11a10547x491S11a18518x507S2c600543x462S2c611522x477 S38900464x493 M526x532S19a20498x488S22e04504x514S36d01479x469 M534x525S10030519x495S10038467x495S2a200505x476S2a218473x476 M537x518S20500497x507S20500463x507S10043507x483S2d600473x493 M551x540S14c57490x513S14c59450x498S20359530x477S20357488x461S26507519x497S26517477x483 M515x511S10047485x490 S38800464x496
    # raw: L536x584S10058480x536S26526506x570S10050498x536S20500493x571S20500526x572S36d03478x517S30a00486x483 L537x518S20500497x507S20500463x507S10043507x483S2d600473x493 L541x531S22e04524x513S15a11518x485S2ff00482x483 L538x553S14c37511x527S14c3f460x527S30d10482x488S30302482x477 S38700463x496 L566x524S10047504x503S2d60a543x503S36d03479x477 L542x524S16d51476x477S16d51522x477S26616459x508S26606501x506S26602501x493S26612459x494 L511x515S10049490x485 L518x553S20320497x447S19220497x466S11502482x489S17620497x508S11920491x527 S38700463x496 M529x530S17620513x495S1f000499x515S36d03479x471 S38700463x496 M567x527S10040523x497S2d608538x479S36d03479x473 M508x515S10e00493x485 M513x582S1f720487x419S18d20487x438S17620492x467S11a20492x487S19220492x521S1fb20493x544S14a20492x567 M532x533S20500522x522S20500498x499S22a05507x508S14210469x467 S38700463x496 M528x593S19220507x442S2a20c488x425S17620507x464S11a20507x483S10120507x517S1f720502x549S11920501x567S36d03479x407 M548x615S2ff00482x483S18710502x505S14c50521x574S14c58477x584S26510481x546S26500526x536S22520516x557S22520473x567 R526x509S14a20475x492S26606496x493 S38700463x496 M564x525S10040520x495S2d608535x477S36d03479x476 M529x538S2e74c472x513S14220498x462S2e700514x497S14228471x478 M531x511S10047469x490S2d60a508x490 M537x549S14c38474x471S14c50492x451S1f738497x517S26703464x481S1f751482x528S26707506x495 M528x523S15a06480x503S15a41473x477S23d04501x489 S38800464x496

    is_valid = raw.startswith('AS') or raw.startswith('M') or raw.startswith('L') or raw.startswith('R')
    if not is_valid:
        return False

    sign = []
    sign_plus = []
    feat_col = []
    feat_row = []
    feat_x = []
    feat_y = []

    for token in raw.split(' '):
        if len(token) > 0:
            if token.startswith('AS'):
                # find first M or L or R box
                MAX = 999999
                IndexOfM = token.find('M')
                IndexOfL = token.find('L')
                IndexOfR = token.find('R')
                IndexOfM = IndexOfM if IndexOfM > 0 else MAX
                IndexOfL = IndexOfL if IndexOfL > 0 else MAX
                IndexOfR = IndexOfR if IndexOfR > 0 else MAX
                startingIndex = min(IndexOfM, IndexOfL, IndexOfR)
                if startingIndex != MAX:
                    token = token[startingIndex:]

            # if not punctuation, add the box information
            if not token.startswith('S'):
                sign.append(token[0])
                sign_plus.append(token[0])
                feat_col.append('0')
                feat_row.append('0')
                feat_x.append(token[1:4])
                feat_y.append(token[5:8])

            # find all symbols
            # how to factorize a symbol: see https://slevinski.github.io/SuttonSignWriting/characters/symbols.html#?ui=en&set=fsw&sym=S100
            for index, ch in enumerate(token):
                if ch == 'S':
                    sign.append(token[index:index + 6])
                    sign_plus.append(token[index:index + 4])
                    feat_col.append(token[index + 4])
                    feat_row.append(token[index + 5])
                    feat_x.append(token[index + 6:index + 9])
                    feat_y.append(token[index + 10:index + 13])

    return ' '.join(sign), ' '.join(sign_plus), ' '.join(feat_col), ' '.join(feat_row), ' '.join(feat_x), ' '.join(feat_y)

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

        sign_sentence = row['sign_writing'].numpy().decode('utf-8')
        if not sign_sentence:
            continue

        # tokenize
        en_tokenized = ' '.join(nltk.word_tokenize(en))

        # run customized sign parser
        parsed = parse(sign_sentence)
        if not parsed:
            continue
        sign, sign_plus, feat_col, feat_row, feat_x, feat_y = parsed

        data_list.append({
            'isDict': isDict, 
            'en': en.encode("unicode_escape").decode("utf-8"),
            'en_tokenized': en_tokenized.encode("unicode_escape").decode("utf-8"),
            'sign': sign,
            'sign+': sign_plus,
            'feat_col': feat_col,
            'feat_row': feat_row,
            'feat_x': feat_x,
            'feat_y': feat_y,  
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

with \
open('./data/train.sign', 'w+') as f_sign, \
open('./data/train.sign+', 'w+') as f_sign_plus, \
open('./data/train.withDict.sign', 'w+') as f_sign_withDict, \
open('./data/train.withDict.sign+', 'w+') as f_sign_plus_withDict, \
open('./data/train.en', 'w+') as f_en, \
open('./data/train.tokenized.en', 'w+') as f_en_tokenized, \
open('./data/train.withDict.en', 'w+') as f_en_withDict, \
open('./data/train.withDict.tokenized.en', 'w+') as f_en_withDict_tokenized, \
open('./data/train.sign+.feat_col', 'w+') as f_feat_col, \
open('./data/train.sign+.feat_row', 'w+') as f_feat_row, \
open('./data/train.sign+.feat_x', 'w+') as f_feat_x, \
open('./data/train.sign+.feat_y', 'w+') as f_feat_y:
    for item in train:
        if not item['isDict']:
            f_en.write("%s\n" % item['en'])
            f_en_tokenized.write("%s\n" % item['en_tokenized'])
            f_sign.write("%s\n" % item['sign'])
            f_sign_plus.write("%s\n" % item['sign+'])
        f_en_withDict.write("%s\n" % item['en'])
        f_en_withDict_tokenized.write("%s\n" % item['en_tokenized'])
        f_sign_withDict.write("%s\n" % item['sign'])
        f_sign_plus_withDict.write("%s\n" % item['sign+'])
        f_feat_col.write("%s\n" % item['feat_col'])
        f_feat_row.write("%s\n" % item['feat_row'])
        f_feat_x.write("%s\n" % item['feat_x'])
        f_feat_y.write("%s\n" % item['feat_y'])
                    
with \
open('./data/dev.sign', 'w+') as f_sign, \
open('./data/dev.sign+', 'w+') as f_sign_plus, \
open('./data/dev.en', 'w+') as f_en, \
open('./data/dev.tokenized.en', 'w+') as f_en_tokenized, \
open('./data/dev.sign+.feat_col', 'w+') as f_feat_col, \
open('./data/dev.sign+.feat_row', 'w+') as f_feat_row, \
open('./data/dev.sign+.feat_x', 'w+') as f_feat_x, \
open('./data/dev.sign+.feat_y', 'w+') as f_feat_y:
    for item in dev:
        f_sign.write("%s\n" % item['sign'])
        f_sign_plus.write("%s\n" % item['sign+'])
        f_en.write("%s\n" % item['en'])
        f_en_tokenized.write("%s\n" % item['en_tokenized'])
        f_feat_col.write("%s\n" % item['feat_col'])
        f_feat_row.write("%s\n" % item['feat_row'])
        f_feat_x.write("%s\n" % item['feat_x'])
        f_feat_y.write("%s\n" % item['feat_y'])

with \
open('./data/test.sign', 'w+') as f_sign, \
open('./data/test.sign+', 'w+') as f_sign_plus, \
open('./data/test.en', 'w+') as f_en, \
open('./data/test.tokenized.en', 'w+') as f_en_tokenized, \
open('./data/test.sign+.feat_col', 'w+') as f_feat_col, \
open('./data/test.sign+.feat_row', 'w+') as f_feat_row, \
open('./data/test.sign+.feat_x', 'w+') as f_feat_x, \
open('./data/test.sign+.feat_y', 'w+') as f_feat_y:
    for item in test:
        f_sign.write("%s\n" % item['sign'])
        f_sign_plus.write("%s\n" % item['sign+'])
        f_en.write("%s\n" % item['en'])
        f_en_tokenized.write("%s\n" % item['en_tokenized'])
        f_feat_col.write("%s\n" % item['feat_col'])
        f_feat_row.write("%s\n" % item['feat_row'])
        f_feat_x.write("%s\n" % item['feat_x'])
        f_feat_y.write("%s\n" % item['feat_y'])