#!/usr/bin/env python3
"""Loopback-only local app. No API key, hosted service, or frontend build needed."""
import argparse
import json
import mimetypes
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from service import Dashboard

ROOT=Path(__file__).resolve().parent

def serve(app,port):
    token=secrets.token_urlsafe(32); lock=threading.RLock()
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,body,kind='application/json'):
            if kind=='application/json': body=json.dumps(body,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers(); self.wfile.write(body)
        def local(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
        def do_GET(self):
            if not self.local(): return self.send(403,{'error':'Local access only'})
            url=urlparse(self.path); q={k:v[0] for k,v in parse_qs(url.query).items()}
            try:
                with lock:
                    if url.path=='/api/state': return self.send(200,dict(app.bootstrap(q.get('learner')),token=token))
                    if url.path=='/api/curve': return self.send(200,app.curve(q['learner'],q['skill'],q.get('task')))
                    if url.path=='/api/export': return self.send(200,app.export(q['learner']))
                files={'/':'index.html','/style.css':'style.css','/app.js':'app.js','/favicon.svg':'favicon.svg'}
                if url.path not in files: return self.send(404,{'error':'Not found'})
                path=ROOT/'static'/files[url.path]
                self.send(200,path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'text/plain')
            except (ValueError,KeyError) as e: self.send(400,{'error':str(e)})
        def do_POST(self):
            if not self.local() or self.headers.get('X-Dashboard-Token')!=token: return self.send(403,{'error':'Reload this local dashboard to continue.'})
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):
                return self.send(403,{'error':'Cross-origin writes are disabled.'})
            if self.headers.get('Content-Type')!='application/json': return self.send(415,{'error':'JSON required'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=40000: return self.send(413,{'error':'Request too large'})
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict): raise ValueError('Expected an object')
                path=urlparse(self.path).path
                if not path.startswith('/api/'): return self.send(404,{'error':'Not found'})
                with lock: result=app.command(path[5:],data)
                self.send(200,result)
            except (ValueError,KeyError,TypeError) as e: self.send(400,{'error':str(e)})
            except Exception:
                import traceback; traceback.print_exc(); self.send(500,{'error':'The operation could not be saved. Your previous records are intact.'})
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--db',type=Path,default=ROOT/'data'/'sessions.sqlite3')
    p.add_argument('--device',choices=('auto','cpu','mps'),default='auto')
    p.add_argument('--max-hints',type=int,default=3); p.add_argument('--max-retries',type=int,default=2)
    args=p.parse_args()
    app=Dashboard(args.db,args.device,args.max_hints,args.max_retries)
    server=serve(app,args.port)
    print(f'\n  RECALL · Python memory lab\n  http://127.0.0.1:{server.server_port}\n  Frozen model: {app.memory.device}\n  Local records: {args.db}\n',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: server.server_close()
