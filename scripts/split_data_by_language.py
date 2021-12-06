import sys

source = 'sign'
target = 'spoken'
model_name = sys.argv[1]

sent_languages = ['en-us', 'pt-br', 'mt-mt']
dict_languages = ['dict.en-us', 'dict.en-sg', 'dict.de-de', 'dict.de-ch', 'dict.fr-ca', 'dict.fr-be', 'dict.fr-ch', 'dict.fr-fr', 'dict.pt-br', 'dict.es-es', 'dict.es-hn', 'dict.es-ar', 'dict.es-ni', 'dict.ca-es', 'dict.ar-tn', 'dict.ko-kr', 'dict.mt-mt', 'dict.nl-be', 'dict.pl-pl', 'dict.sk-sk', 'dict.sl-sl']
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
        language_name = tags[0][2:4] + '-' + tags[1][2:4]
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