import sys
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

st = StanfordNERTagger('./tools/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
					   './tools/stanford-ner/stanford-ner.jar',
					   encoding='utf-8')

hyps_path = sys.argv[1]
gold_path = sys.argv[2]

with open(hyps_path) as hyps_file, \
     open(gold_path) as target_file:

    hyps_lines = [line.rstrip() for line in hyps_file.readlines()]
    gold_lines = [line.rstrip() for line in target_file.readlines()]

    total_count = 0
    hit_count = 0

    for i, gold_line in enumerate(gold_lines):
        hyps_line = hyps_lines[i]

        gold_classified = st.tag(word_tokenize(gold_line))
        hyps_classified = st.tag(word_tokenize(hyps_line))

        gold_tokens = [token for token, entity in gold_classified if entity != 'O']
        hyps_tokens = [token for token, entity in hyps_classified if entity != 'O']

        print('line {}'.format(i + 1))
        print(gold_tokens)
        print(hyps_tokens)

        total_count += len(gold_tokens)
        hit_count += len([token for token in gold_tokens if token in hyps_tokens])
    
    print(hit_count / total_count)
        
