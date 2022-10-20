import re
import random
import math

import tensorflow_datasets as tfds
import sign_language_datasets.datasets


random.seed(42)

if __name__ == "__main__":
    signbank = tfds.load(name='sign_bank')['train']
    stats = {}

    # see https://github.com/sign-language-processing/datasets/blob/master/sign_language_datasets/datasets/signbank/signbank.py#L164
    for index, row in enumerate(signbank):
        puddle_id = row['puddle'].numpy().item()
        assumed_spoken_language_code = row['assumed_spoken_language_code'].numpy().decode('utf-8')
        country_code = row['country_code'].numpy().decode('utf-8')
        terms = [f.decode('utf-8') for f in row['terms'].numpy()]
        sign_sentence = row['sign_writing'].numpy().decode('utf-8')

        if not puddle_id or not assumed_spoken_language_code or not country_code or \
            not sign_sentence or len(terms) < 1:
            continue

        spoken = ''
        is_dict = False

        # Sentences
        # en-us: Literature US, ASL Bible Books NLT, ASL Bible Books Shores Deaf Church
        # pt-br: Literatura Brasil
        if puddle_id == 5 or puddle_id == 151 or puddle_id == 152 or \
        puddle_id == 114:
            # the first element is the title
            # the second is the main text
            if len(terms) > 1:
                # only take the main text
                spoken = terms[1]

                if puddle_id == 151:
                    # remove line-break and source, e.g., Nehemiah 3v11 NLT
                    spoken = re.sub(r"\n\n.*NLT", "", spoken)

        # Dictionaries
        # en-us: Dictionary US, LLCN & SignTyp, ASL Bible Dictionary
        # de-de: Wörterbuch DE
        # fr-ca: Dictionnaire Quebec
        # pt-br: Dicionário Brasil, Enciclopédia Brasil
        if puddle_id == 4 or puddle_id == 25 or puddle_id == 28 or \
        puddle_id == 53 or \
        puddle_id == 47 or \
        puddle_id == 46 or puddle_id == 116:
            # longest word allowed
            if len(terms[0]) < 100:
                spoken = terms[0]
            is_dict = True

        if not spoken or spoken.startswith('<iframe'):
            continue

        # escape newline to \n
        spoken = spoken.replace("\n", "\\n")

        text_length = len(spoken.split(' '))

        if puddle_id in stats:
            stats[puddle_id]['count'] += 1
            stats[puddle_id]['text_length_sum'] += text_length
        else:
            stats[puddle_id] = {
                'count': 1,
                'text_length_sum': text_length,
            }

        # if index > 100:
        #     break

    for puddle_id in stats:
        stats[puddle_id]['text_length_mean'] = stats[puddle_id]['text_length_sum'] / stats[puddle_id]['count']

    from pprint import pprint
    pprint(stats)