import sys

source = 'sign'
target = 'spoken'
model_name = sys.argv[1] or 'baseline_multilingual'

sent_languages = ['en', 'pt']
dict_languages = ['dict.fr', 'dict.de', 'dict.en', 'dict.pt']
languages = sent_languages + dict_languages

n_best = 5

filedata = {filename: open('./data/test.{}'.format(filename), 'w+') for filename in languages}
filedata_hyps = {filename: open('./models/{}/best.hyps.test.{}'.format(model_name, filename), 'w+') for filename in languages}

with open('./data/test.{}'.format(source)) as source_file, \
     open('./data/test.{}'.format(target)) as target_file, \
     open('./models/{}/best.hyps.test.raw'.format(model_name)) as hyps_file:

    source_lines = [line.rstrip() for line in source_file.readlines()]
    target_lines = [line.rstrip() for line in target_file.readlines()]
    hyps_lines = [line.rstrip() for line in hyps_file.readlines()]

    for index, source_line in enumerate(source_lines):
        tags = source_line.split(' ')[:3]
        language_name = tags[0][2:4]
        is_dict = tags[2] == '<dict>'
        filename = 'dict.{}'.format(language_name) if is_dict else language_name

        # write gold samples
        file = filedata[filename]
        file.write("%s\n" % target_lines[index])

        if filename in sent_languages:
            # write the top-1 hyps for bleu/chrf
            file_hyps = filedata_hyps[filename]
            file_hyps.write("%s\n" % hyps_lines[index * n_best])
        elif filename in dict_languages:
            # write the top-n hyps for top-n accuracy
            file_hyps = filedata_hyps[filename]
            for i in range(n_best):
                file_hyps.write("%s\n" % hyps_lines[index * n_best + i])

for file in filedata.values():
    file.close()
for file in filedata_hyps.values():
    file.close()