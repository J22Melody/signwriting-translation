import sys
from itertools import zip_longest

with open(sys.argv[1]) as file_hyps, open(sys.argv[2]) as file_gold:
    total_distance = 0
    total_number = 0

    hyps_lines = [line.rstrip() for line in file_hyps.readlines()]
    gold_lines = [line.rstrip() for line in file_gold.readlines()]

    assert len(hyps_lines) == len(gold_lines)

    for hyps_line, gold_line in zip(hyps_lines, gold_lines):

        hyps_numbers = hyps_line.split(' ')
        gold_numbers = gold_line.split(' ')

        zipped = list(zip_longest(hyps_numbers, gold_numbers, fillvalue=482))

        total_number += len(zipped)

        for hyps, gold in zipped:
            distance = abs(int(hyps) - int(gold))
            total_distance += distance

    mean_distance = total_distance / total_number

    print('total distance:', total_distance)
    print('token number:', total_number)
    print('mean distance:', mean_distance)
        
        