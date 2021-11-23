import numpy as np

length_list = []

with open('./data/train.sign') as file:
    for line in file:
        length = len(line.rstrip().split(' '))
        length_list.append(length)

print('avg length: ', sum(length_list) / len(length_list))
print('max length: ', max(length_list))
print('99.5th percentile length: ', np.percentile(np.array(length_list), 99.5))

length_list = []

with open('./data/train.spoken') as file:
    for line in file:
        length = len(line.rstrip().split(' '))
        length_list.append(length)

print('avg length: ', sum(length_list) / len(length_list))
print('max length: ', max(length_list))
print('99.9th percentile length: ', np.percentile(np.array(length_list), 99.9))