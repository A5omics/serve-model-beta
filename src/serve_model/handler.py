"""Vercel function. Only a SHA-256 digest of the high-entropy bearer key is bundled."""
from functools import lru_cache
from hashlib import sha256
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import json

ASSETS = Path(__file__).parent / "_model"

@lru_cache(maxsize=1)
def engine():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ASSETS / "tokenizer.json"))
    tok.enable_truncation(max_length=512)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    return tok, ort.InferenceSession(str(ASSETS / "model.onnx"), opts, providers=["CPUExecutionProvider"])

class handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Cache-Control","no-store")
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(405,{"error":"POST required"})

    def do_POST(self):
        if self.path != "/predict":
            return self.reply(404,{"error":"unknown route"})
        header = self.headers.get("Authorization","")
        digest = (ASSETS/"key.sha256").read_text().strip()
        if not header.startswith("Bearer ") or not compare_digest(sha256(header[7:].encode()).hexdigest(),digest):
            return self.reply(401,{"error":"invalid key"})
        try:
            size = int(self.headers.get("Content-Length","0"))
            if not 0 < size <= 32768:
                return self.reply(413,{"error":"invalid body size"})
            text = json.loads(self.rfile.read(size))["text"]
            if not isinstance(text,str) or not text.strip():
                raise ValueError()
        except (TypeError, ValueError, KeyError):
            return self.reply(400,{"error":"expected a nonempty text string"})
        import numpy as np
        tok, session = engine()
        ids = tok.encode(text).ids
        scores = session.run(None,{"input_ids":np.array([ids],dtype=np.int64)})[0][0]
        labels = json.loads((ASSETS/"labels.json").read_text())
        self.reply(200,{"label":labels[int(scores.argmax())],"scores":dict(zip(labels,map(float,scores)))})

if __name__ == "__main__":
    import os
    from http.server import ThreadingHTTPServer
    engine()  # Fail startup before accepting requests if the artifact is invalid.
    if "SERVE_LISTEN_FD" in os.environ:
        import socket
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler, bind_and_activate=False)
        server.socket.close()
        server.socket = socket.socket(fileno=int(os.environ["SERVE_LISTEN_FD"]))
        server.server_address = server.socket.getsockname()
    else:
        server = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), handler)
    server.serve_forever()
