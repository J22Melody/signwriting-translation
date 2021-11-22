source = 'sign'
target = 'spoken'

languages = ['dict.fr', 'en', 'pt', 'dict.de', 'dict.en', 'dict.pt']

filedata = {filename: open('./data/test.{}'.format(filename), 'w+', encoding='UTF8') for filename in languages}

with open('./data/test.{}'.format(source)) as source_file, \
     open('./data/test.{}'.format(target)) as target_file:
    source_lines = [line.rstrip() for line in source_file.readlines()]
    target_lines = [line.rstrip() for line in target_file.readlines()]

    for index, source_line in enumerate(source_lines):
        tags = source_line.split(' ')[:3]
        language_name = tags[0][2:4]
        is_dict = tags[2] == '<dict>'
        filename = 'dict.{}'.format(language_name) if is_dict else language_name
        file = filedata[filename]
        file.write("%s\n" % target_lines[index])

for file in filedata.values():
    file.close()