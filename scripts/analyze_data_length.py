import numpy as np
import sys

length_list = []

with open(sys.argv[1]) as file:
    for line in file:
        length = len(line.rstrip().split(' '))
        length_list.append(length)

print('avg length: ', sum(length_list) / len(length_list))
print('max length: ', max(length_list))
print('95th percentile length: ', np.percentile(np.array(length_list), 95))