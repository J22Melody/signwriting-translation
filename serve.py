"""Run app.py's Flask app, serving only sign2spoken.

MXNet is retired (no installable wheels left) and app.py only uses it for the
spoken2sign direction, so stub it before the import. app.run() in app.py's
__main__ binds 127.0.0.1, hence this entrypoint (binds 0.0.0.0 for Docker).
"""

import os
import sys
import types

mxnet = types.ModuleType("mxnet")
mxnet.cpu = lambda: None
sockeye = types.ModuleType("sockeye")
sockeye.model = types.ModuleType("sockeye.model")
sockeye.model.load_models = lambda **kwargs: ([], [], [])
sockeye.inference = types.ModuleType("sockeye.inference")
sys.modules.update({
    "mxnet": mxnet,
    "sockeye": sockeye,
    "sockeye.model": sockeye.model,
    "sockeye.inference": sockeye.inference,
})

from app import app  # noqa: E402

app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3030)), threaded=False)
