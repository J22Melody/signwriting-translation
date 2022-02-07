This project requires Python 3.

Install requirements as follows:

`pip install -r requirements.txt`

Run with gunicorn as follows:

`gunicorn -w 4 -b 0.0.0.0:5000 app:app`