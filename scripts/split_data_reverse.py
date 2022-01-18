import sys

source = 'spoken'
target = 'symbol'
model_name = sys.argv[1]

languages = ['en', 'pt', 'de', 'fr']

filedata_sym = {filename: open('./data_reverse/test.{}.symbol'.format(filename), 'w+') for filename in languages}
filedata_hyps_sym = {filename: open('./models/{}/test.hyps.{}.symbol'.format(model_name, filename), 'w+') for filename in languages}

with open('./data_reverse/test.{}'.format(source)) as source_file, \
     open('./data_reverse/test.{}'.format(target)) as target_file, \
     open('./models/{}/test.hyps.symbol'.format(model_name)) as hyps_file:

    source_lines = [line.rstrip() for line in source_file.readlines()]
    target_lines = [line.rstrip() for line in target_file.readlines()]
    hyps_lines = [line.rstrip() for line in hyps_file.readlines()]

    for index, source_line in enumerate(source_lines):
        tags = source_line.split(' ')[:3]
        language_name = tags[0][2:4]
        filename = language_name

        # write gold samples
        line = target_lines[index]
        filedata_sym[filename].write("%s\n" % line)

        # write hyps samples
        line = hyps_lines[index]
        filedata_hyps_sym[filename].write("%s\n" % line)

for file in filedata_sym.values():
    file.close()
for file in filedata_hyps_sym.values():
    file.close()