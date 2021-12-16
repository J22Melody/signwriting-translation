import sys

default_x = '482'
default_y = '483'

with open(sys.argv[1]) as file_symbol, \
     open(sys.argv[2]) as file_number, \
     open(sys.argv[3], 'w+') as file_fsw:
    symbol_lines = [line.rstrip() for line in file_symbol.readlines()]
    number_lines = [line.rstrip() for line in file_number.readlines()]

    assert len(symbol_lines) == len(number_lines)

    for index, symbol_line in enumerate(symbol_lines):
        number_line = number_lines[index]
        symbols = [symbol for symbol in symbol_line.split(' ') if symbol != 'P']
        numbers = number_line.split(' ')

        expected_numbers_len = len(symbols) * 2       
        if expected_numbers_len != len(numbers):
            offset = expected_numbers_len - len(numbers)
            print('Warning: line {} not aligned by {}'.format(index + 1, offset))

        fsw_line = ''
        for i, symbol in enumerate(symbols):
            x_i = 2 * i
            y_i = 2 * i + 1

            if not symbol.startswith('S') or symbol.startswith('S387') or symbol.startswith('S388') or \
            symbol.startswith('S389') or symbol.startswith('S38a') or symbol.startswith('S38b'):
                fsw_line += ' '
            
            fsw_line += symbol
            fsw_line += numbers[x_i] if x_i < len(numbers) else default_x
            fsw_line += 'x'
            fsw_line += numbers[y_i] if y_i < len(numbers) else default_y

        file_fsw.write("%s\n" % fsw_line)