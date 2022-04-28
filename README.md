# signwriting-translation

This repo contains code and documentation for training bilingual and multilingual translation models between spoken languages and signed languages in [SignWriting](https://www.signwriting.org/). We also provide an API server for inferring based on the best trained models.

## Environment

We have the following main dependencies:

- We use Python 3.8.11 (a virtual Python environment is recommended).
- We use [sign-language-datasets](https://github.com/sign-language-processing/datasets) as our training data source.
- We use [SentencePiece](https://github.com/google/sentencepiece) for tokenization.
- We use [joeynmt](https://github.com/joeynmt/joeynmt) for training models translating from signed languages to spoken languages (SIGN to SPOKEN).
- We use [Sockeye](https://github.com/awslabs/sockeye) for training models translating from spoken languages to signed languages (SPOKEN to SIGN), which in turn depends on [MXNet](mxnet.apache.org).


Install common dependencies:

`pip install -r requirements.txt`

`sudo apt-get install sentencepiece`

For SIGN to SPOKEN, install `joeynmt` (a customized version supporting source factors):

`pip install git+ssh://git@github.com/J22Melody/joeynmt.git@factors_complete`

or 

`pip install <path_to_local_joeynmt>`

For SPOKEN to SIGN, install `MXNet` and `Sockeye`:

`pip install --pre -f https://dist.mxnet.io/python 'mxnet>=2.0.0b2021'`

`pip install sockeye==3.0.13`

Note `joeynmt` and `Sockeye` require different versions of the package `sacreBLEU`, thus results in version conflict if both installed. 

For training, we recommend two separate environments for the two different settings. For inferring (using the API server), this conflict can be safely ignored because `sacreBLEU` is not used there.

## Model Training

### 40K SIGN to SPOKEN (EN-US) (Bilingual)

### 100K SIGN to SPOKEN (Multilingual)

### 100K SPOKEN to SIGN (Multilingual)

## API Server

Run locally for debugging:

`python app.py`

Run with `Gunicorn` for deployment:

`gunicorn -w 4 -b 0.0.0.0:3030 app:app`

Example `Supervisor` config file (`/etc/supervisor/conf.d/gunicorn.conf`):

```
[program:gunicorn]
user=xxx
directory=/home/xxx/signwriting-translation
command=gunicorn -w 4 -b 0.0.0.0:3030 app:app

autostart=true
autorestart=true
stdout_logfile=/home/xxx/log/gunicorn.log
stderr_logfile=/home/xxx/log/gunicorn.err.log
```

See [API_spec.md](https://github.com/J22Melody/signwriting-translation/blob/main/API_spec.md) for API specifications.