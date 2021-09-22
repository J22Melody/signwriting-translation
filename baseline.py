import json
import re
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow_datasets as tfds
import sign_language_datasets.datasets

# ASL Bible Books NLT
class MyDataset(Dataset):
    def __init__(self, fromLocalFile=True):
        self.data_list = []

        if fromLocalFile:
            with open('bible.txt') as file:
                for line in file:
                    self.data_list.append(json.loads(line.rstrip()))
        else:
            signbank = tfds.load(name='sign_bank')['train']

            for index, row in enumerate(signbank):  
                if row['puddle'].numpy().item() == 151:
                    terms = [f.decode('utf-8') for f in row['terms'].numpy()]

                    if len(terms) == 2:
                        self.data_list.append({
                            'spoken': re.sub(r"\n\n.*NLT", "", terms[1]),
                            # TODO: parse sign_writing by https://github.com/sutton-signwriting/core
                            'sign': row['sign_writing'].numpy().decode('utf-8'),
                        })

            with open('bible.txt', 'w') as f:
                for item in self.data_list:
                    f.write("%s\n" % json.dumps(item))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

training_data = MyDataset(False)
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

for row in train_dataloader:
    print(row)
    exit()