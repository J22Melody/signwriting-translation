This project requires Python 3.

Install requirements as follows:

`sudo apt-get install sentencepiece`

`pip install --pre -f https://dist.mxnet.io/python 'mxnet>=2.0.0b2021'`

`pip install -r requirements.txt`

Run with gunicorn as follows:

`gunicorn -w 4 -b 0.0.0.0:3030 app:app`