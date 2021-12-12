import sys
from collections.abc import Iterable

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def parse(line):
    tokens = line.split(' ')
    numbers = [token for token in tokens if token.isnumeric()]
    symbols = [token for token in tokens if not token.isnumeric() and token != 'x']

    # order-invariant: sort symbols within a sign
    # signs = []
    # current = None
    # for token in symbols:
    #     if token.startswith('S'):
    #         current.append(token)
    #     else:
    #         current = []
    #         signs.append(token)
    #         signs.append(current)
    # signs = [sorted(sign) if isinstance(sign, list) else sign for sign in signs]
    # symbols = flatten(signs)

    return ' '.join(numbers), ' '.join(symbols) 

source = 'spoken'
target = 'sign'
model_name = sys.argv[1]

file_all_num = open('./models/{}/best.hyps.test.num.sign'.format(model_name), 'w+')
file_all_sym = open('./models/{}/best.hyps.test.sym.sign'.format(model_name), 'w+')

sent_languages = ['en', 'pt']
dict_languages = ['dict.fr', 'dict.de', 'dict.en', 'dict.pt']
languages = sent_languages + dict_languages

filedata = {filename: open('./data_reverse/test.{}.sign'.format(filename), 'w+') for filename in languages}
filedata_hyps = {filename: open('./models/{}/best.hyps.test.{}.sign'.format(model_name, filename), 'w+') for filename in languages}
filedata_num = {filename: open('./data_reverse/test.{}.num.sign'.format(filename), 'w+') for filename in languages}
filedata_hyps_num = {filename: open('./models/{}/best.hyps.test.{}.num.sign'.format(model_name, filename), 'w+') for filename in languages}
filedata_sym = {filename: open('./data_reverse/test.{}.sym.sign'.format(filename), 'w+') for filename in languages}
filedata_hyps_sym = {filename: open('./models/{}/best.hyps.test.{}.sym.sign'.format(model_name, filename), 'w+') for filename in languages}

with open('./data_reverse/test.{}'.format(source)) as source_file, \
     open('./data_reverse/test.{}'.format(target)) as target_file, \
     open('./models/{}/best.hyps.test'.format(model_name)) as hyps_file:

    source_lines = [line.rstrip() for line in source_file.readlines()]
    target_lines = [line.rstrip() for line in target_file.readlines()]
    hyps_lines = [line.rstrip() for line in hyps_file.readlines()]

    for index, source_line in enumerate(source_lines):
        tags = source_line.split(' ')[:3]
        language_name = tags[0][2:4]
        is_dict = tags[2] == '<dict>'
        filename = 'dict.{}'.format(language_name) if is_dict else language_name

        # write gold samples
        line = target_lines[index]
        numbers, symbols = parse(line)
        filedata[filename].write("%s\n" % line)
        filedata_num[filename].write("%s\n" % numbers)
        filedata_sym[filename].write("%s\n" % symbols)

        # write hyps samples
        line = hyps_lines[index]
        numbers, symbols = parse(line)
        file_all_num.write("%s\n" % numbers)
        file_all_sym.write("%s\n" % symbols)
        filedata_hyps[filename].write("%s\n" % line)
        filedata_hyps_num[filename].write("%s\n" % numbers)
        filedata_hyps_sym[filename].write("%s\n" % symbols)

file_all_num.close()
file_all_sym.close()
for file in filedata.values():
    file.close()
for file in filedata_hyps.values():
    file.close()