import subprocess
import json
import sys
from os import environ
from flask import Flask, request
from flask_cors import CORS

import mxnet as mx
from sockeye import inference, model

from scripts.fetch_data import parse

MODEL_PATH = './models'
CONFIG_PATH = './configs'

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/translate/<direction>', methods=['POST'])
def translate(direction):
    payload = request.get_json()
    language_code = payload.get('language_code', 'en')
    country_code = payload.get('country_code', 'us')
    translation_type = payload.get('translation_type', 'sent')
    text = payload.get('text', '')
    translations = []

    if direction == 'sign2spoken':
        n_best = payload.get('n_best', '5')

        model_name = 'baseline_multilingual_plus'
        config_path = '{}/{}.yaml'.format(CONFIG_PATH, model_name)
        model_path = '{}/{}/best.ckpt'.format(MODEL_PATH, model_name)
        spm_path = './data_plus/spm.model'

        parsed = parse(text)
        inputs = parsed[1:-1]
        # inputs = ['S33100', 'S331', '0', '0', '500', '500', '0', '0']
        
        inputs = ['<2{}> <4{}> <{}> {}'.format(language_code, country_code, translation_type, input) if i < 2
            else '-1 -1 -1 {}'.format(input) for i, input in enumerate(inputs)]
        input_str = '|'.join(inputs)

        command = 'echo "{}" | {} -m joeynmt translate {} -n {} --ckpt {} | spm_decode --model={}'.format(
            input_str, sys.executable, config_path, n_best, model_path, spm_path)
        output = subprocess.run(command, shell=True, check=True, capture_output=True)
        translations = output.stdout.decode("utf-8").split('\n')[:-1]

    elif direction == 'spoken2sign':
        n_best = int(payload.get('n_best', '3'))
        beam_size = n_best

        spm_path = './data_reverse/spm.model'
        tag_str = '<2{}> <4{}> <{}>'.format(language_code, country_code, translation_type)
        command = 'echo "{} {}" | spm_encode --model={}'.format(tag_str, text, spm_path)
        input = subprocess.run(command, shell=True, check=True, capture_output=True)
        input = input.stdout.decode("utf-8")
        
        translator = inference.Translator(context=context,
                                          ensemble_mode='linear',
                                          scorer=inference.CandidateScorer(),
                                          output_scores=True,
                                          batch_size=1,
                                          beam_size=beam_size,
                                          beam_search_stop='all',
                                          nbest_size=n_best,
                                          models=sockeye_models,
                                          source_vocabs=sockeye_source_vocabs,
                                          target_vocabs=sockeye_target_vocabs)

        input = inference.make_input_from_plain_string(0, input)
        output = translator.translate([input])[0]

        symbols_candidates = output.nbest_translations
        factors_candidates = output.nbest_factor_translations
        for symbols, factors in zip(symbols_candidates, factors_candidates):
            symbols = symbols.split(' ')
            xs = factors[0].split(' ')
            ys = factors[1].split(' ')
            fsw = ''

            for i, (symbol, x, y) in enumerate(zip(symbols, xs, ys)):
                if symbol != 'P':
                    if i != 0:
                        if not symbol.startswith('S') or symbol.startswith('S387') or symbol.startswith('S388') or \
                        symbol.startswith('S389') or symbol.startswith('S38a') or symbol.startswith('S38b'):
                            fsw += ' '
                    fsw += symbol
                    fsw += x
                    fsw += 'x'
                    fsw += y

            translations.append(fsw)

    return {
        'direction': direction,
        'language_code': language_code,
        'country_code': country_code,
        'translation_type': translation_type,
        'n_best': n_best,
        'text': text,
        'translations': translations,
        # 'translations': [{
        #     'text': text,
        #     'perplexity': 0,
        # }],
    }

if __name__ == '__main__':
    port = int(environ.get('PORT', 5000))
    with app.app_context():
        global context
        context = mx.cpu()

        global sockeye_models, sockeye_source_vocabs, sockeye_target_vocabs
        model_name = 'sockeye_spoken2symbol_factor_0.1'
        model_path = '{}/{}'.format(MODEL_PATH, model_name)
        sockeye_models, sockeye_source_vocabs, sockeye_target_vocabs = model.load_models(
            context=context, dtype=None, model_folders=[model_path], inference_only=True)

        app.run(threaded=False,
                debug=False,
                port=port)