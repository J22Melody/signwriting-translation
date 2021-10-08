import time
import math
import json
import re
import random
import subprocess
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import nltk
nltk.download('punkt')

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.legacy.data import BucketIterator

import tensorflow_datasets as tfds
import sign_language_datasets.datasets


SAMPLE_LIMIT = 999999
N_EPOCH = 10
VAL_EVERY = 32
EARLY_STOPPING_N = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset and Dataloader
def parse(raw, includeSpatials=False):
    sequence = []

    # if not punctuation, add the box information
    if includeSpatials and not raw.startswith('S'):
        for ch2 in raw[0:8]:
            sequence.append(ch2)

    for index, ch in enumerate(raw):
        if ch == 'S':
            # sequence of symbols
            sequence.append(raw[index:index + 6])

            # spatials
            if includeSpatials:
                for ch2 in raw[index + 6:index + 13]:
                    sequence.append(ch2)

    return ' '.join(sequence)

# ASL Bible Books NLT
class MyDataset(Dataset):
    def __init__(self, fromLocalFile=True, limit=9999999):
        self.data_list = []

        if fromLocalFile:
            with open('bible.txt') as file:
                for index, line in enumerate(file):
                    if index == limit:
                        break

                    self.data_list.append(json.loads(line.rstrip()))
        else:
            signbank = tfds.load(name='sign_bank')['train']

            for index, row in enumerate(signbank):
                puddle_id = row['puddle'].numpy().item()
                if puddle_id == 151 or puddle_id == 152:
                    terms = [f.decode('utf-8') for f in row['terms'].numpy()]

                    # the first element is chapter
                    # the second is the main text
                    if len(terms) == 2:
                        # only take the main text
                        en = terms[1].lower()
                        # remove line-break and source, e.g., Nehemiah 3v11 NLT
                        en = re.sub(r"\n\n.*NLT", "", en)
                        # tokenize
                        en = ' '.join(nltk.word_tokenize(en))

                        sign = row['sign_writing'].numpy().decode('utf-8')

                        # run standard js parser (https://github.com/sutton-signwriting/core/blob/master/src/fsw/fsw-parse.js#L63)
                        # FIXME: js parser not compatible with dataset input
                        # result = subprocess.run(['node', 'parse.js', sign], stdout=subprocess.PIPE)
                        # sign = result.stdout

                        # run customized parser
                        signs = sign.split(' ')
                        signs = list(map(parse, signs))
                        sign = ' '.join(signs)

                        self.data_list.append({
                            'en': en,
                            'sign': sign,
                        })

            with open('bible.txt', 'w') as f:
                for item in self.data_list:
                    f.write("%s\n" % json.dumps(item))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return [item['sign'], item['en']]

dataset = MyDataset(fromLocalFile=True, limit=SAMPLE_LIMIT)
total_len = len(dataset)
print('total size', total_len)

train_len = math.floor(total_len*0.9)
val_len = math.floor(total_len*0.05)
test_len = total_len - train_len - val_len
train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(SEED))
print('train size:', len(train_set))
print('val size:', len(val_set))
print('test size:', len(test_set))

sort_key = lambda x: len(x[1])
train_dataloader = BucketIterator(train_set, batch_size=BATCH_SIZE, shuffle=True, sort_key=sort_key, sort=False, sort_within_batch=True, train=True)
val_dataloader = BucketIterator(val_set, batch_size=BATCH_SIZE, shuffle=True, sort_key=sort_key, sort=False, sort_within_batch=True, train=False)
test_dataloader = BucketIterator(test_set, batch_size=BATCH_SIZE, shuffle=True, sort_key=sort_key, sort=False, sort_within_batch=True, train=False)


# Dictionary
SOS_token = 0
EOS_token = 1
PADDING_token = 2

class Lang:
    def __init__(self, name, char_level=False):
        self.name = name
        self.char_level = char_level
        self.word2index = {"PADDING": 2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PADDING"}
        self.n_words = 3  # Count SOS and EOS
        self.max_length = 0

    def tokenize(self, sentence):
        if self.char_level:
            return sentence
        else:
            return sentence.split(' ')

    def addSentence(self, sentence):
        sentence = self.tokenize(sentence)

        for word in sentence:
            self.addWord(word)
        
        length = len(sentence) + 1
        if length > self.max_length:
            self.max_length = length

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

input_lang = Lang('sign', char_level=False)
output_lang = Lang('en', char_level=False)

for row in dataset:
    input_lang.addSentence(row[0])
    output_lang.addSentence(row[1])

print('input vocab size:', input_lang.n_words)
print('output vocab size:', output_lang.n_words)
print('input max len:', input_lang.max_length)
print('output max len:', output_lang.max_length)


# Seq2seq model
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        embedded = self.embedding(input).view(batch_size, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, input_lang.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        batch_size = input.size(0)
        embedded = self.embedding(input).view(batch_size, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden.repeat(batch_size, 1, 1)), 2)), dim=2)
        # cut attn_weights to match sequence length
        attn_weights_matched = attn_weights[:, :, :encoder_outputs.size(1)]
        attn_applied = torch.bmm(attn_weights_matched, encoder_outputs)

        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output), dim=2).squeeze(1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# Helpers
def indexesFromSentence(lang, sentence, max_length):
    return [
        lang.word2index[sentence[i]] if i < len(sentence) else 
        (PADDING_token if i > len(sentence) else EOS_token)
        for i in range(max_length)
    ]

def tensorFromSentence(lang, sentences):
    sentences = list(map(lang.tokenize, sentences))
    max_length = len(max(sentences, key=len)) + 1
    indexes = []
    for s in sentences:
        idx = indexesFromSentence(lang, s, max_length)
        indexes.append(idx)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(len(sentences), max_length, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# Training
teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder.train()
    decoder.train()

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)

    batch_size = input_tensor.size(0)

    encoder_outputs = torch.zeros(batch_size, input_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[:, ei], encoder_hidden)
        encoder_outputs[:, ei] = encoder_output[:, 0]

    decoder_input = torch.tensor([[SOS_token]] * batch_size, device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[:, di].view(-1))
            decoder_input = target_tensor[:, di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[:, di].view(-1))

            # TODO: how to do this in a batch?
            # if decoder_input.item() == EOS_token:
            #     break

            # FIXME: temporary workaround - fall back to teacher forcing
            for i in range(batch_size):
                if decoder_input[i].item() == EOS_token:
                    decoder_input[i] = target_tensor[i, di]

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_epoch=1, print_every=1, val_every=8, plot_every=100, learning_rate=0.0001, early_stopping_n=10):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    n_iters = n_epoch * train_len / BATCH_SIZE
    iter = 0

    early_stopping_count = 0
    last_val_loss_avg = 99999999

    for i in range(n_epoch):
        print('Epoch', i + 1)

        train_dataloader.create_batches()

        for index, pair in enumerate(train_dataloader.batches):
            iter = iter + 1

            # transpose list, see https://stackoverflow.com/questions/6473679/transpose-list-of-lists
            pair = list(map(list, zip(*pair)))

            pair = tensorsFromPair(pair)
            input_tensor = pair[0]
            target_tensor = pair[1]

            # print(target_tensor.shape)
            # print(target_tensor.view(BATCH_SIZE, -1))
            # continue

            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('Train step: %d (%.2f%%) / time: %s / loss: %.4f' % 
                    (iter, iter / n_iters * 100, timeSince(start, iter / n_iters), print_loss_avg))

            if iter % val_every == 0:
                val_loss_avg = validate(encoder, decoder, criterion)
                print('Val step: %d / loss: %.4f' % (iter / val_every, val_loss_avg))

                # early stopping
                if val_loss_avg > last_val_loss_avg:
                    early_stopping_count += 1
                    if early_stopping_count == early_stopping_n:
                        print('early stop!')
                        return
                last_val_loss_avg = val_loss_avg

                evaluateRandomly(encoder, decoder, n=1)

            # if iter % plot_every == 0:
            #     plot_loss_avg = plot_loss_total / plot_every
            #     plot_losses.append(plot_loss_avg)
            #     plot_loss_total = 0

    # showPlot(plot_losses)


# Validation
def validate(encoder, decoder, criterion):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        loss_total = 0
        count = 0

        val_dataloader.create_batches()

        for index, pair in enumerate(val_dataloader.batches):
            # transpose list, see https://stackoverflow.com/questions/6473679/transpose-list-of-lists
            pair = list(map(list, zip(*pair)))

            pair = tensorsFromPair(pair)
            input_tensor = pair[0]
            target_tensor = pair[1]

            input_length = input_tensor.size(1)
            target_length = target_tensor.size(1)
            batch_size = input_tensor.size(0)

            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(batch_size, input_length, encoder.hidden_size, device=device)

            loss = 0

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[:, ei], encoder_hidden)
                encoder_outputs[:, ei] = encoder_output[:, 0]

            decoder_input = torch.tensor([[SOS_token]] * batch_size, device=device)
            decoder_hidden = encoder_hidden

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[:, di].view(-1))

                # TODO: how to do this in a batch?
                # if decoder_input.item() == EOS_token:
                #     break

                # FIXME: temporary workaround - fall back to teacher forcing
                for i in range(batch_size):
                    if decoder_input[i].item() == EOS_token:
                        decoder_input[i] = target_tensor[i, di]
            
            loss_total += loss.item() / target_length
            count = index + 1

        return loss_total / count


# Evaluation
def evaluate(encoder, decoder, sentence):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, [sentence])
        input_length = input_tensor.size()[1]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(1, input_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[:, ei], encoder_hidden)
            encoder_outputs[:, ei] = encoder_output[:, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(1, output_lang.max_length, input_lang.max_length)

        for di in range(output_lang.max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[0, di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi[0].item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            elif topi[0].item() == PADDING_token:
                decoded_words.append('[]')
            else:
                decoded_words.append(output_lang.index2word[topi[0].item()])

            decoder_input = topi.detach()

        return decoded_words, decoder_attentions[0, :di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(list(test_set))
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words).strip()
        print('<', output_sentence)
        print('')


# Main steps: train, save, load, eval
PATH = 'baseline4.pt'
def train_and_save():
    encoder1 = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
    attn_decoder1 = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, n_epoch=N_EPOCH, learning_rate=LEARNING_RATE, val_every=VAL_EVERY, early_stopping_n=EARLY_STOPPING_N)
    
    torch.save({
        'encoder': encoder1.state_dict(),
        'decoder': attn_decoder1.state_dict(),
    }, PATH)

def load_and_eval():
    encoder1 = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
    attn_decoder1 = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words, dropout_p=0.1).to(device)

    checkpoint = torch.load(PATH)
    encoder1.load_state_dict(checkpoint['encoder'])
    attn_decoder1.load_state_dict(checkpoint['decoder'])

    evaluateRandomly(encoder1, attn_decoder1)

train_and_save()
load_and_eval()