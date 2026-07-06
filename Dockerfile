# Serves the sign2spoken API (app.py) with the joeynmt baseline_multilingual_plus
# model. The spoken2sign direction is not included: it needs MXNet, which is
# retired and no longer installable; see the Sockeye models on HuggingFace
# (https://huggingface.co/sign/sockeye-text-to-factored-signwriting) instead.
#
# linux/amd64: torch 1.10 has no arm64 linux wheels.
FROM --platform=linux/amd64 python:3.9-slim

# git: pip install from GitHub. sentencepiece: app.py shells out to spm_decode.
RUN apt-get update && apt-get install -y --no-install-recommends git sentencepiece \
    && rm -rf /var/lib/apt/lists/*

# Serving needs flask + the custom joeynmt fork (source-factor support) with its
# matching torch/torchtext pair (CPU wheels, pinned by requirements.txt).
# numpy<2 and setuptools==59.5.0: torch 1.10 is built against the numpy 1.x C
# API and the pre-59.6 distutils layout (also pinned by requirements.txt).
RUN pip install --no-cache-dir torch==1.10.0+cpu torchtext==0.11.0 \
        -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir flask flask-cors gdown "numpy<2" "setuptools==59.5.0" \
        "git+https://github.com/J22Melody/joeynmt.git@factors_complete"

# torch 1.10's libtorch requests an executable stack, which emulation (running
# this amd64 image on an arm64 host) refuses to map.
COPY clear_execstack.py /
RUN python /clear_execstack.py /usr/local/lib/python3.9/site-packages/torch/lib/*.so*

WORKDIR /app
COPY . .

# Model checkpoints live on Google Drive, not in the repo (see the README).
RUN gdown 1HS_yB4lp1893u-e9Tr2XvMKOZcd33Q70 -O models/baseline_multilingual_plus/best.ckpt

ENV PORT=3030
EXPOSE 3030
CMD ["python", "serve.py"]
