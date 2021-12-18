import sys

model_name = sys.argv[1]

with open('./models/{}/test.hyps.mixed'.format(model_name)) as source_file, \
     open('./models/{}/test.hyps.symbol'.format(model_name), 'w+') as symbol_file, \
     open('./models/{}/test.hyps.feat_x'.format(model_name), 'w+') as x_file, \
     open('./models/{}/test.hyps.feat_y'.format(model_name), 'w+') as y_file, \
     open('./models/{}/test.hyps.fsw'.format(model_name), 'w+') as fsw_file:

    source_lines = [line.rstrip() for line in source_file.readlines()]

    for index, source_line in enumerate(source_lines):
        symbols = []
        xs = []
        ys = []
        fsw_line = ''

        for i, item in enumerate(source_line.split(' ')):
            symbol, x, y = item.split('|')

            symbols.append(symbol)
            xs.append(x)
            ys.append(y)

            if symbol != 'P':
                if i != 0:
                    if not symbol.startswith('S') or symbol.startswith('S387') or symbol.startswith('S388') or \
                    symbol.startswith('S389') or symbol.startswith('S38a') or symbol.startswith('S38b'):
                        fsw_line += ' '
                fsw_line += symbol
                fsw_line += x
                fsw_line += 'x'
                fsw_line += y

        symbol_file.write("%s\n" % ' '.join(symbols))
        x_file.write("%s\n" % ' '.join(xs))
        y_file.write("%s\n" % ' '.join(ys))
        fsw_file.write("%s\n" % fsw_line)