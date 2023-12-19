import argparse
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from scripts.fetch_data import parse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
args = parser.parse_args()

data_list = []
for item in open(args.input, 'r').readlines():
    item = item.strip()
    tags = ' '.join(item.split(' ')[:2]) + ' '
    fsw = ' '.join(item.split(' ')[2:])

    parsed = parse(fsw)
    if not parsed:
        print(f'cannot parse {fsw}')
        continue

    _, sign, sign_plus, feat_col, feat_row, feat_x, feat_y, feat_x_rel, feat_y_rel, _ = parsed

    # add language, country, dict tag on source side
    sign = tags + sign
    sign_plus = tags + sign_plus
    tags_feat = '-1 -1 '
    feat_col = tags_feat + feat_col
    feat_row = tags_feat + feat_row
    feat_x = tags_feat + feat_x
    feat_y = tags_feat + feat_y
    feat_x_rel = tags_feat + feat_x_rel
    feat_y_rel = tags_feat + feat_y_rel

    data_list.append({
        # sign2spoken
        'sign': sign,
        'sign+': sign_plus,
        'feat_col': feat_col,
        'feat_row': feat_row,
        'feat_x': feat_x,
        'feat_y': feat_y,  
        'feat_x_rel': feat_x_rel,
        'feat_y_rel': feat_y_rel,  
    })

# print(data_list)

prefix = args.input.replace('.fsw', '')
with \
open(f'{prefix}.sign', 'w+') as f_sign, \
open(f'{prefix}.sign+', 'w+') as f_sign_plus, \
open(f'{prefix}.feat_col', 'w+') as f_feat_col, \
open(f'{prefix}.feat_row', 'w+') as f_feat_row, \
open(f'{prefix}.feat_x', 'w+') as f_feat_x, \
open(f'{prefix}.feat_y', 'w+') as f_feat_y, \
open(f'{prefix}.feat_x_rel', 'w+') as f_feat_x_rel, \
open(f'{prefix}.feat_y_rel', 'w+') as f_feat_y_rel:
    for item in data_list:
        f_sign.write("%s\n" % item['sign'])
        f_sign_plus.write("%s\n" % item['sign+'])
        f_feat_col.write("%s\n" % item['feat_col'])
        f_feat_row.write("%s\n" % item['feat_row'])
        f_feat_x.write("%s\n" % item['feat_x'])
        f_feat_y.write("%s\n" % item['feat_y'])
        f_feat_x_rel.write("%s\n" % item['feat_x_rel'])
        f_feat_y_rel.write("%s\n" % item['feat_y_rel'])

