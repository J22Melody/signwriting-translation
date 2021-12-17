with open('./models/sockeye_spoken2symbol_factor/test.hyps.mixed') as source_file, \
     open('./models/sockeye_spoken2symbol_factor/test.hyps.symbol', 'w+') as symbol_file, \
     open('./models/sockeye_spoken2symbol_factor/test.hyps.fsw', 'w+') as fsw_file:

    source_lines = [line.rstrip() for line in source_file.readlines()]

    for index, source_line in enumerate(source_lines):
        symbols = []
        fsw_line = ''

        for i, item in enumerate(source_line.split(' ')):
            symbol, x, y = item.split('|')

            symbols.append(symbol)

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
        fsw_file.write("%s\n" % fsw_line)