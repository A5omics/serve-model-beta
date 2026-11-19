import http.client
import json
from pathlib import Path
import subprocess
import time

image = "serve-model-public-beta-smoke:0.1.0b1"
root = Path.cwd()
subprocess.run(["docker","build","-t",image,str(root/"container")],check=True)
cid=subprocess.check_output(["docker","run","-d","--rm","-p","127.0.0.1::8080",image],text=True).strip()
try:
    port = int(subprocess.check_output(["docker","port",cid,"8080/tcp"],text=True).strip().split(":")[-1])
    key=(Path.home()/".config/a5-serve-model/local/local/smoke-pilot/api-key").read_text()
    def call(message,key=None):
        conn=http.client.HTTPConnection("127.0.0.1",port,timeout=3)
        headers={"Content-Type":"application/json"}
        if key:
            headers["Authorization"]="Bearer "+key
        try:
            conn.request("POST","/predict",json.dumps({"text":message}),headers)
            response=conn.getresponse()
            return response.status,json.load(response)
        finally:
            conn.close()
    deadline=time.monotonic()+60
    while True:
        try:
            assert call("test")[0]==401
            break
        except OSError:
            if time.monotonic()>deadline:
                raise
            time.sleep(0.5)
    assert call("test","incorrect")[0]==401
    results={}
    for text,label in [("Please send my invoice","billing"),("The dashboard crashes whenever I open it","technical")]:
        status,body=call(text,key)
        assert status==200 and body["label"]==label,(status,body)
        results[text]=body
    print(json.dumps({"anonymous":401,"wrong_key":401,"predictions":results},indent=2))
finally:
    subprocess.run(["docker","stop",cid],check=True,stdout=subprocess.DEVNULL)
