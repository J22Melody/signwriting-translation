import re
import random
import math

import tensorflow_datasets as tfds
import sign_language_datasets.datasets


random.seed(42)

def parse(raw):
    # raw: AS10011S10019S2e704S2e748M525x535S2e748483x510S10011501x466S2e704510x500S10019476x475
    # raw: M528x518S15a07472x487S1f010490x503S26507515x483 M509x515S18720491x486 S38800464x496 M522x557S18518458x542S30d00478x483S36d01480x517S18510488x542 S38700463x496 M531x529S15a56487x517S1fb20490x495S20600509x502S36d01479x472 M515x511S10047485x490 M537x522S1f750510x507S1f758474x507S2d201511x478S2d210464x478 M541x508S17620481x492S26512459x494S17628503x492S26506526x494 M562x537S2ff00482x483S11a10547x491S11a18518x507S2c600543x462S2c611522x477 S38900464x493 M526x532S19a20498x488S22e04504x514S36d01479x469 M534x525S10030519x495S10038467x495S2a200505x476S2a218473x476 M537x518S20500497x507S20500463x507S10043507x483S2d600473x493 M551x540S14c57490x513S14c59450x498S20359530x477S20357488x461S26507519x497S26517477x483 M515x511S10047485x490 S38800464x496
    # raw: L536x584S10058480x536S26526506x570S10050498x536S20500493x571S20500526x572S36d03478x517S30a00486x483 L537x518S20500497x507S20500463x507S10043507x483S2d600473x493 L541x531S22e04524x513S15a11518x485S2ff00482x483 L538x553S14c37511x527S14c3f460x527S30d10482x488S30302482x477 S38700463x496 L566x524S10047504x503S2d60a543x503S36d03479x477 L542x524S16d51476x477S16d51522x477S26616459x508S26606501x506S26602501x493S26612459x494 L511x515S10049490x485 L518x553S20320497x447S19220497x466S11502482x489S17620497x508S11920491x527 S38700463x496 M529x530S17620513x495S1f000499x515S36d03479x471 S38700463x496 M567x527S10040523x497S2d608538x479S36d03479x473 M508x515S10e00493x485 M513x582S1f720487x419S18d20487x438S17620492x467S11a20492x487S19220492x521S1fb20493x544S14a20492x567 M532x533S20500522x522S20500498x499S22a05507x508S14210469x467 S38700463x496 M528x593S19220507x442S2a20c488x425S17620507x464S11a20507x483S10120507x517S1f720502x549S11920501x567S36d03479x407 M548x615S2ff00482x483S18710502x505S14c50521x574S14c58477x584S26510481x546S26500526x536S22520516x557S22520473x567 R526x509S14a20475x492S26606496x493 S38700463x496 M564x525S10040520x495S2d608535x477S36d03479x476 M529x538S2e74c472x513S14220498x462S2e700514x497S14228471x478 M531x511S10047469x490S2d60a508x490 M537x549S14c38474x471S14c50492x451S1f738497x517S26703464x481S1f751482x528S26707506x495 M528x523S15a06480x503S15a41473x477S23d04501x489 S38800464x496

    is_valid = raw.startswith('AS') or raw.startswith('M') or raw.startswith('L') or raw.startswith('R') or raw.startswith('B')
    if not is_valid:
        return False

    sign = []
    sign_plus = []
    feat_col = []
    feat_row = []
    feat_x = []
    feat_y = []
    feat_x_rel = []
    feat_y_rel = []
    sign_reverse = []

    for token in raw.split(' '):
        if len(token) > 0:
            if token.startswith('AS'):
                # find first box marker
                MAX = 999999
                IndexOfM = token.find('M')
                IndexOfL = token.find('L')
                IndexOfR = token.find('R')
                IndexOfB = token.find('B')
                IndexOfM = IndexOfM if IndexOfM > 0 else MAX
                IndexOfL = IndexOfL if IndexOfL > 0 else MAX
                IndexOfR = IndexOfR if IndexOfR > 0 else MAX
                IndexOfB = IndexOfB if IndexOfB > 0 else MAX
                startingIndex = min(IndexOfM, IndexOfL, IndexOfR, IndexOfB)
                if startingIndex != MAX:
                    token = token[startingIndex:]

            # if not punctuation (without box marker), add the box marker
            if not token.startswith('S'):
                sign.append(token[0])
                sign_plus.append(token[0])
                feat_col.append('-1')
                feat_row.append('-1')
                feat_x.append(token[1:4])
                feat_y.append(token[5:8])
                feat_x_rel.append('-1')
                feat_y_rel.append('-1')
                sign_reverse.append(token[0])
                sign_reverse.append(token[1:4])
                sign_reverse.append(token[5:8])
            else:
                sign.append('P')
                sign_plus.append('P')
                feat_col.append('-1')
                feat_row.append('-1')
                feat_x.append('-1')
                feat_y.append('-1')
                feat_x_rel.append('-1')
                feat_y_rel.append('-1')
                sign_reverse.append('P')

            # find all symbols
            # how to factorize a symbol: see https://slevinski.github.io/SuttonSignWriting/characters/symbols.html#?ui=spoken&set=fsw&sym=S100
            symbols = []
            for index, ch in enumerate(token):
                if ch == 'S':
                    symbols.append({
                        'sign': token[index:index + 6],
                        'sign_plus': token[index:index + 4],
                        'feat_col': token[index + 4],
                        'feat_row': token[index + 5],
                        'x': token[index + 6:index + 9],
                        'y': token[index + 10:index + 13],
                    })

            x_sorted = list(dict.fromkeys(sorted([int(s['x']) for s in symbols])))
            y_sorted = list(dict.fromkeys(sorted([int(s['y']) for s in symbols])))

            for s in symbols:
                sign.append(s['sign'])
                sign_plus.append(s['sign_plus'])
                feat_col.append(s['feat_col'])
                feat_row.append(s['feat_row'])
                feat_x.append(s['x'])
                feat_y.append(s['y'])
                feat_x_rel.append(str(x_sorted.index(int(s['x']))))
                feat_y_rel.append(str(y_sorted.index(int(s['y']))))
                sign_reverse.append(s['sign'])
                sign_reverse.append(s['x'])
                sign_reverse.append(s['y'])

    return raw, ' '.join(sign), ' '.join(sign_plus), ' '.join(feat_col), ' '.join(feat_row), ' '.join(feat_x), ' '.join(feat_y), ' '.join(feat_x_rel), ' '.join(feat_y_rel), ' '.join(sign_reverse)

signbank = tfds.load(name='sign_bank')['train']
data_list = []

# see https://github.com/sign-language-processing/datasets/blob/master/sign_language_datasets/datasets/signbank/signbank.py#L164
for index, row in enumerate(signbank):
    puddle_id = row['puddle'].numpy().item()
    assumed_spoken_language_code = row['assumed_spoken_language_code'].numpy().decode('utf-8')
    country_code = row['country_code'].numpy().decode('utf-8')
    terms = [f.decode('utf-8') for f in row['terms'].numpy()]
    sign_sentence = row['sign_writing'].numpy().decode('utf-8')

    if not puddle_id or not assumed_spoken_language_code or not country_code or \
        not sign_sentence or len(terms) < 1:
        continue

    spoken = ''
    is_dict = False

    # Sentences
    # en-us: Literature US, ASL Bible Books NLT, ASL Bible Books Shores Deaf Church
    # pt-br: Literatura Brasil
    if puddle_id == 5 or puddle_id == 151 or puddle_id == 152 or \
       puddle_id == 114:
        # the first element is the title
        # the second is the main text
        if len(terms) > 1:
            # only take the main text
            spoken = terms[1]

            if puddle_id == 151:
                # remove line-break and source, e.g., Nehemiah 3v11 NLT
                spoken = re.sub(r"\n\n.*NLT", "", spoken)

    # Dictionaries
    # en-us: Dictionary US, LLCN & SignTyp, ASL Bible Dictionary
    # de-de: Wörterbuch DE
    # fr-ca: Dictionnaire Quebec
    # pt-br: Dicionário Brasil, Enciclopédia Brasil
    if puddle_id == 4 or puddle_id == 25 or puddle_id == 28 or \
       puddle_id == 53 or \
       puddle_id == 47 or \
       puddle_id == 46 or puddle_id == 116:
        # longest word allowed
        if len(terms[0]) < 100:
            spoken = terms[0]
        is_dict = True

    if not spoken or spoken.startswith('<iframe'):
        continue

    # escape newline to \n
    spoken = spoken.replace("\n", "\\n")

    # run customized sign parser
    parsed = parse(sign_sentence)
    if not parsed:
        continue

    fsw, sign, sign_plus, feat_col, feat_row, feat_x, feat_y, feat_x_rel, feat_y_rel, sign_reverse = parsed

    # add language, country, dict tag on source side
    tags = '<2{}> <4{}> <{}> '.format(assumed_spoken_language_code, country_code, 'dict' if is_dict else 'sent')
    sign = tags + sign
    sign_plus = tags + sign_plus
    tags_feat = '-1 -1 -1 '
    feat_col = tags_feat + feat_col
    feat_row = tags_feat + feat_row
    feat_x = tags_feat + feat_x
    feat_y = tags_feat + feat_y
    feat_x_rel = tags_feat + feat_x_rel
    feat_y_rel = tags_feat + feat_y_rel
    spoken_reverse = tags + spoken

    data_list.append({
        'fsw': fsw,
        # sign2spoken
        'spoken': spoken, 
        'sign': sign,
        'sign+': sign_plus,
        'feat_col': feat_col,
        'feat_row': feat_row,
        'feat_x': feat_x,
        'feat_y': feat_y,  
        'feat_x_rel': feat_x_rel,
        'feat_y_rel': feat_y_rel,  
        # spoken2sign
        'spoken_reverse': spoken_reverse,
        'sign_reverse': sign_reverse,
        'symbol_reverse': ' '.join([token for token in sign_reverse.split(' ') if not token.isnumeric()]),
        'number_reverse': ' '.join([token for token in sign_reverse.split(' ') if token.isnumeric()]),
    })

    print(index)
    # if index > 100:
    #     break

random.shuffle(data_list)

# sign2spoken

total_size = len(data_list)
train_size = math.floor(total_size*0.95)
dev_size = math.floor(total_size*0.03)
test_size = total_size - train_size - dev_size

dev = data_list[:dev_size]
test = data_list[dev_size:test_size + dev_size]
train = data_list[test_size + dev_size:]

random.shuffle(train)

with \
open('./data/train.sign', 'w+') as f_sign, \
open('./data/train.sign+', 'w+') as f_sign_plus, \
open('./data/train.spoken', 'w+') as f_spoken, \
open('./data/train.feat_col', 'w+') as f_feat_col, \
open('./data/train.feat_row', 'w+') as f_feat_row, \
open('./data/train.feat_x', 'w+') as f_feat_x, \
open('./data/train.feat_y', 'w+') as f_feat_y, \
open('./data/train.feat_x_rel', 'w+') as f_feat_x_rel, \
open('./data/train.feat_y_rel', 'w+') as f_feat_y_rel:
    for item in train:
        f_spoken.write("%s\n" % item['spoken'])
        f_sign.write("%s\n" % item['sign'])
        f_sign_plus.write("%s\n" % item['sign+'])
        f_feat_col.write("%s\n" % item['feat_col'])
        f_feat_row.write("%s\n" % item['feat_row'])
        f_feat_x.write("%s\n" % item['feat_x'])
        f_feat_y.write("%s\n" % item['feat_y'])
        f_feat_x_rel.write("%s\n" % item['feat_x_rel'])
        f_feat_y_rel.write("%s\n" % item['feat_y_rel'])
                    
with \
open('./data/dev.sign', 'w+') as f_sign, \
open('./data/dev.sign+', 'w+') as f_sign_plus, \
open('./data/dev.spoken', 'w+') as f_spoken, \
open('./data/dev.feat_col', 'w+') as f_feat_col, \
open('./data/dev.feat_row', 'w+') as f_feat_row, \
open('./data/dev.feat_x', 'w+') as f_feat_x, \
open('./data/dev.feat_y', 'w+') as f_feat_y, \
open('./data/dev.feat_x_rel', 'w+') as f_feat_x_rel, \
open('./data/dev.feat_y_rel', 'w+') as f_feat_y_rel:
    for item in dev:
        f_sign.write("%s\n" % item['sign'])
        f_sign_plus.write("%s\n" % item['sign+'])
        f_spoken.write("%s\n" % item['spoken'])
        f_feat_col.write("%s\n" % item['feat_col'])
        f_feat_row.write("%s\n" % item['feat_row'])
        f_feat_x.write("%s\n" % item['feat_x'])
        f_feat_y.write("%s\n" % item['feat_y'])
        f_feat_x_rel.write("%s\n" % item['feat_x_rel'])
        f_feat_y_rel.write("%s\n" % item['feat_y_rel'])

with \
open('./data/test.sign', 'w+') as f_sign, \
open('./data/test.sign+', 'w+') as f_sign_plus, \
open('./data/test.spoken', 'w+') as f_spoken, \
open('./data/test.feat_col', 'w+') as f_feat_col, \
open('./data/test.feat_row', 'w+') as f_feat_row, \
open('./data/test.feat_x', 'w+') as f_feat_x, \
open('./data/test.feat_y', 'w+') as f_feat_y, \
open('./data/test.feat_x_rel', 'w+') as f_feat_x_rel, \
open('./data/test.feat_y_rel', 'w+') as f_feat_y_rel:
    for item in test:
        f_sign.write("%s\n" % item['sign'])
        f_sign_plus.write("%s\n" % item['sign+'])
        f_spoken.write("%s\n" % item['spoken'])
        f_feat_col.write("%s\n" % item['feat_col'])
        f_feat_row.write("%s\n" % item['feat_row'])
        f_feat_x.write("%s\n" % item['feat_x'])
        f_feat_y.write("%s\n" % item['feat_y'])
        f_feat_x_rel.write("%s\n" % item['feat_x_rel'])
        f_feat_y_rel.write("%s\n" % item['feat_y_rel'])

# spoken2sign

train_size = math.floor(total_size*0.98)
dev_size = math.floor(total_size*0.01)
test_size = total_size - train_size - dev_size

dev = data_list[:dev_size]
test = data_list[dev_size:test_size + dev_size]
train = data_list[test_size + dev_size:]

random.shuffle(train)

with \
open('./data_reverse/train.sign', 'w+') as f_sign, \
open('./data_reverse/train.symbol', 'w+') as f_symbol, \
open('./data_reverse/train.number', 'w+') as f_number, \
open('./data_reverse/train.fsw', 'w+') as f_fsw, \
open('./data_reverse/train.spoken', 'w+') as f_spoken:
    for item in train:
        f_sign.write("%s\n" % item['sign_reverse'])
        f_symbol.write("%s\n" % item['symbol_reverse'])
        f_number.write("%s\n" % item['number_reverse'])
        f_fsw.write("%s\n" % item['fsw'])
        f_spoken.write("%s\n" % item['spoken_reverse'])
                    
with \
open('./data_reverse/dev.sign', 'w+') as f_sign, \
open('./data_reverse/dev.symbol', 'w+') as f_symbol, \
open('./data_reverse/dev.number', 'w+') as f_number, \
open('./data_reverse/dev.fsw', 'w+') as f_fsw, \
open('./data_reverse/dev.spoken', 'w+') as f_spoken:
    for item in dev:
        f_sign.write("%s\n" % item['sign_reverse'])
        f_symbol.write("%s\n" % item['symbol_reverse'])
        f_number.write("%s\n" % item['number_reverse'])
        f_fsw.write("%s\n" % item['fsw'])
        f_spoken.write("%s\n" % item['spoken_reverse'])

with \
open('./data_reverse/test.sign', 'w+') as f_sign, \
open('./data_reverse/test.symbol', 'w+') as f_symbol, \
open('./data_reverse/test.number', 'w+') as f_number, \
open('./data_reverse/test.fsw', 'w+') as f_fsw, \
open('./data_reverse/test.spoken', 'w+') as f_spoken:
    for item in test:
        f_sign.write("%s\n" % item['sign_reverse'])
        f_symbol.write("%s\n" % item['symbol_reverse'])
        f_number.write("%s\n" % item['number_reverse'])
        f_fsw.write("%s\n" % item['fsw'])
        f_spoken.write("%s\n" % item['spoken_reverse'])