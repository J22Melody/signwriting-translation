import re
import random
import math

train = []

# fingerspelling data generation
fingerspelling_map = {
    'A': 'S1f720',
    'B': 'S14720',
    'C': 'S16d20',
    'D': 'S10120',
    'E': 'S14a20',
    'F': 'S1ce20',
    'G': 'S1f000',
    'H': 'S11502',
    'I': 'S19220',
    'J': ['S19220', 'S2a20c'],
    'K': 'S14020',
    'L': 'S1dc20',
    'M': 'S18d20',
    'N': 'S11920',
    'O': 'S17620',
    'P': 'S14051',
    'Q': 'S1f051',
    'R': 'S11a20',
    'S': 'S20320',
    'T': 'S1fb20',
    'U': 'S11520',
    'V': 'S10e20',
    'W': 'S18620',
    'X': 'S10620',
    'Y': 'S19a20',
    'Z': ['S10020', 'S2450a'],
}

for _ in range(10000):
    word_len = random.randrange(10) + 1

    spoken = ''
    sign = '<2en> <4us> <fngr> M'
    sign_plus = '<2en> <4us> <fngr> M'
    feat_col = '-1 -1 -1 -1'
    feat_row = '-1 -1 -1 -1'
    feat_x = '-1 -1 -1 539'
    feat_y = '-1 -1 -1 542'
    feat_x_rel = '-1 -1 -1 -1'
    feat_y_rel = '-1 -1 -1 -1'

    base_x = 490
    base_y = 456

    for i in range(word_len):
        letter, symbols = random.choice(list(fingerspelling_map.items()))
        spoken += letter.lower() if i != 0 else letter
        for symbol in symbols if isinstance(symbols, list) else [symbols]:
            sign += ' {}'.format(symbol)
            sign_plus += ' {}'.format(symbol[:4])
            feat_col += ' {}'.format(symbol[4])
            feat_row += ' {}'.format(symbol[5])
            feat_x += ' {}'.format(base_x)
            feat_y += ' {}'.format(base_y)
            feat_x_rel += ' {}'.format(0)
            feat_y_rel += ' {}'.format(i)
            base_y += 15

    train.append({
        'spoken': spoken,
        'sign': sign,
        'sign+': sign_plus,
        'feat_col': feat_col,
        'feat_row': feat_row,
        'feat_x': feat_x,
        'feat_y': feat_y,  
        'feat_x_rel': feat_x_rel,
        'feat_y_rel': feat_y_rel,  
    })

with \
open('./data/train.sign', 'a') as f_sign, \
open('./data/train.sign+', 'a') as f_sign_plus, \
open('./data/train.spoken', 'a') as f_spoken, \
open('./data/train.feat_col', 'a') as f_feat_col, \
open('./data/train.feat_row', 'a') as f_feat_row, \
open('./data/train.feat_x', 'a') as f_feat_x, \
open('./data/train.feat_y', 'a') as f_feat_y, \
open('./data/train.feat_x_rel', 'a') as f_feat_x_rel, \
open('./data/train.feat_y_rel', 'a') as f_feat_y_rel:
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
                    