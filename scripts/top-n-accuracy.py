import sys
from collections import defaultdict

n_best = 5

hyps_path = sys.argv[1]
gold_path = sys.argv[2]

with open(hyps_path) as hyps_file, \
     open(gold_path) as target_file:

    hyps_lines = [line.rstrip() for line in hyps_file.readlines()]
    gold_lines = [line.rstrip() for line in target_file.readlines()]

    correct_counts = defaultdict(int)

    for i, gold_line in enumerate(gold_lines):
        for j in range(n_best):
            start_index = i * n_best
            offset = j + 1
            candidates = hyps_lines[start_index:start_index+offset]
            if gold_line in candidates:
                correct_counts[offset] += 1

    total = len(gold_lines)
    for i in range(1, n_best + 1):
        print('top {} accuracy: {}'.format(i, correct_counts[i] / total))
