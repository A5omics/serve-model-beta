import argparse
import hashlib
import sys
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
import urllib.error
import urllib.request

from .export import ROOT, export

SAMPLE = "Please refund the second charge"
DOCKERFILE = """FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy==2.2.6 onnxruntime==1.22.1 tokenizers==0.21.4
COPY handler.py .
COPY _model ./_model
USER 65534
CMD ["python", "handler.py"]
"""


def bundle(adapter, target, key):
    assets = target / "_model"
    assets.mkdir()
    expected = export(adapter, assets)
    (assets / "key.sha256").write_text(hashlib.sha256(key.encode()).hexdigest())
    shutil.copyfile(ROOT / "handler.py", target / "handler.py")
    (target / "Dockerfile").write_text(DOCKERFILE)
    return expected


def check_url(url, key, expected):
    body = json.dumps({"text": SAMPLE}).encode()
    for authenticated in (False, True):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(url, body, headers)
        # Prevent the bearer from following an unexpected redirect.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args):
                return None
        try:
            with urllib.request.build_opener(NoRedirect).open(req, timeout=120) as response:
                actual = json.load(response)
        except urllib.error.HTTPError as error:
            if not authenticated and error.code == 401:
                continue
            raise
        if not authenticated:
            raise SystemExit("Anonymous inference unexpectedly succeeded.")
        if actual.get("label") != max(expected, key=expected.get) or any(
            abs(actual["scores"][label] - score) > 1e-5 for label, score in expected.items()
        ):
            raise SystemExit("Remote prediction differs from the local model.")


@contextmanager
def local_endpoint(stage):
    # Pass a pre-bound loopback socket so the OS can allocate a free port safely.
    import socket
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        process = subprocess.Popen(
            [sys.executable, str(stage/"handler.py")],
            env={**os.environ, "SERVE_LISTEN_FD": str(listener.fileno())},
            pass_fds=(listener.fileno(),), stdout=sys.stderr,
        )
        try:
            yield f"http://127.0.0.1:{port}/predict", process
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()


def main():
    p = argparse.ArgumentParser(prog="serve-model")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("sample", help="Print the bundled sample adapter path")
    for name in ("deploy", "bundle", "local"):
        sub = commands.add_parser(name)
        sub.add_argument("adapter", help="Path to your adapter or 'sample'")
        sub.add_argument("--json", action="store_true", help="Print a machine-readable result on stdout")
        sub.add_argument("--service", default="serve-model")
        if name == "deploy":
            sub.add_argument("--project", required=True, help="Your Google Cloud project ID")
            sub.add_argument("--region", default="us-central1")
            sub.add_argument("--domain", help="Existing HTTPS hostname already routed to this service")
        elif name == "bundle":
            sub.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.command == "sample":
        print(ROOT / "sample")
        return
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}[a-z0-9]", args.service):
        p.error("Use a 2–49 character lowercase service name.")
    project = getattr(args, "project", "local")
    region = getattr(args, "region", "local")
    if args.command == "deploy":
        if not shutil.which("gcloud"):
            p.error("Install Google Cloud CLI and run gcloud auth login.")
        if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]",project) or not re.fullmatch(r"[a-z]+-[a-z]+[0-9]+",region):
            p.error("Invalid project ID or region.")
        if args.domain and not re.fullmatch(r"[a-z0-9.-]+",args.domain):
            p.error("Supply a hostname without a scheme or path.")
    adapter = ROOT / "sample" if args.adapter == "sample" else Path(args.adapter).resolve()
    state = Path.home()/".config"/"a5-serve-model"/project/region/args.service
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    keypath = state/"api-key"
    try:
        with os.fdopen(os.open(keypath,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),"w") as f:
            f.write(secrets.token_urlsafe(48))
    except FileExistsError:
        pass
    key = keypath.read_text()
    if len(key)<40:
        raise SystemExit("Incomplete local key file.")
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp)
        expected = bundle(adapter,stage,key)
        if args.command == "local":
            with local_endpoint(stage) as (endpoint, process):
                check_url(endpoint, key, expected)
                receipt = {"action":"local","endpoint":endpoint,"key_file":str(keypath),
                           "parity_verified":True,"anonymous_status":401}
                print(json.dumps(receipt), flush=True)
                try:
                    returncode = process.wait()
                except KeyboardInterrupt:
                    return
                if returncode:
                    raise SystemExit(returncode)
                return
        if args.command == "bundle":
            shutil.copytree(stage,args.out)
            result = {"action":"bundle","directory":str(args.out.resolve())}
        else:
            subprocess.run(["gcloud","run","deploy",args.service,"--project",project,"--region",region,"--source",str(stage),
                            "--allow-unauthenticated","--min-instances","0","--max-instances","2","--memory","1Gi","--concurrency","4","--quiet"],check=True,stdout=sys.stderr)
            origin = subprocess.check_output(["gcloud","run","services","describe",args.service,"--project",project,
                           "--region",region,"--format=value(status.url)"],text=True).strip()
            if not origin.startswith("https://") or not origin.endswith(".run.app"):
                raise SystemExit("Unexpected Cloud Run origin")
            check_url(origin+"/predict", key, expected)
            endpoint = origin+"/predict"
            if args.domain:
                endpoint = "https://"+args.domain+"/predict"
                check_url(endpoint,key,expected)
            result = {"action":"deploy","endpoint":endpoint,"parity_verified":True,"anonymous_status":401}
    result["key_file"] = str(keypath)
    print(json.dumps(result) if args.json else "\n".join(f"{k}: {v}" for k,v in result.items()))
