import sys
import math

default_x = '482'
default_y = '483'

with open(sys.argv[1]) as file_hyps, open(sys.argv[2]) as file_gold:
    hyps_lines = [line.rstrip() for line in file_hyps.readlines()]
    gold_lines = [line.rstrip() for line in file_gold.readlines()]

    assert len(hyps_lines) == len(gold_lines)

    for index, hyps_line in enumerate(hyps_lines):
        gold_line = gold_lines[index]
        hyps_numbers = hyps_line.split(' ')
        gold_numbers = gold_line.split(' ')
