"""HTTP server and CLI entrypoint for 3/adaptive_web/."""

from __future__ import annotations

import argparse
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from typing import Any
import urllib.parse

from .app import AdaptiveLearningApp, InvalidActionError
from .config import Settings

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 1024 * 1024  # 1 MiB


class _AdaptiveHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64


def _reject_non_finite(val: str) -> Any:
    raise ValueError(f"Non-finite JSON constant not allowed: {val}")


def _is_local_host(host_header: str | None) -> bool:
    if not host_header:
        return False
    host = host_header.split(":")[0].strip().lower()
    return host in ("127.0.0.1", "localhost", "[::1]")


def _is_local_origin(origin_header: str | None) -> bool:
    if not origin_header:
        return True
    parsed = urllib.parse.urlparse(origin_header)
    return parsed.scheme in ("http", "https") and (parsed.hostname or "").lower() in (
        "127.0.0.1",
        "localhost",
        "::1",
    )


def serve(app: AdaptiveLearningApp, port: int = 8780) -> ThreadingHTTPServer:
    """Create and return a configured ThreadingHTTPServer for AdaptiveLearningApp."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_security_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
            )

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self._send_security_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, filename: str, content_type: str) -> None:
            target = (STATIC_DIR / filename).resolve()
            if not target.is_file() or STATIC_DIR.resolve() not in target.parents:
                self._send_json(404, {"error": "Static asset not found."})
                return
            body = target.read_bytes()
            self.send_response(200)
            self._send_security_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _check_host(self) -> bool:
            if not _is_local_host(self.headers.get("Host")):
                self._send_json(403, {"error": "Forbidden Host header."})
                return False
            return True

        def do_GET(self) -> None:
            if not self._check_host():
                return

            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            if ".." in path:
                self._send_json(403, {"error": "Path traversal forbidden."})
                return

            if path == "/":
                self._send_static("index.html", "text/html; charset=utf-8")
                return
            if path == "/style.css":
                self._send_static("style.css", "text/css; charset=utf-8")
                return
            if path == "/app.js":
                self._send_static("app.js", "application/javascript; charset=utf-8")
                return
            if path == "/favicon.svg":
                self._send_static("favicon.svg", "image/svg+xml; charset=utf-8")
                return

            try:
                if path == "/api/state":
                    learner = qs.get("learner", [None])[0]
                    skill = qs.get("skill", [None])[0]
                    self._send_json(200, app.get_state(learner=learner, skill=skill))
                    return
                if path == "/api/curve":
                    learner = qs.get("learner", [None])[0]
                    skill = qs.get("skill", [None])[0]
                    fmt = qs.get("format", ["output_prediction"])[0]
                    task = qs.get("task", [None])[0]
                    self._send_json(200, app.curve(learner, skill, fmt=fmt, task=task))
                    return
                if path == "/api/export":
                    learner = qs.get("learner", [None])[0]
                    self._send_json(200, app.export(learner))
                    return
            except (ValueError, KeyError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(500, {"error": f"Internal error: {exc}"})
                return

            self._send_json(404, {"error": f"Unknown route: {path}"})

        def do_POST(self) -> None:
            if not self._check_host():
                return

            if not _is_local_origin(self.headers.get("Origin")):
                self._send_json(403, {"error": "Forbidden Origin header."})
                return

            csrf = self.headers.get("X-CSRF-Token") or self.headers.get("X-Dashboard-Token")
            if csrf != app.csrf_token:
                self._send_json(403, {"error": "Missing or invalid CSRF token."})
                return

            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                self._send_json(415, {"error": "Content-Type must be application/json."})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length."})
                return

            if length <= 0:
                self._send_json(400, {"error": "Empty request body."})
                return
            if length > MAX_BODY_BYTES:
                if length <= 16 * 1024 * 1024:
                    self.rfile.read(length)
                self._send_json(413, {"error": "Request body exceeds 1 MiB limit."})
                return

            raw_body = self.rfile.read(length)
            try:
                data = json.loads(raw_body.decode("utf-8"), parse_constant=_reject_non_finite)
            except Exception as exc:
                self._send_json(400, {"error": f"Malformed JSON: {exc}"})
                return

            if not isinstance(data, dict):
                self._send_json(400, {"error": "JSON body must be an object."})
                return

            path = urllib.parse.urlparse(self.path).path
            req_id = data.get("request_id")

            try:
                if path == "/api/profile":
                    self._send_json(200, app.create_profile(data.get("name", ""), request_id=req_id))
                    return
                if path == "/api/select_skill":
                    self._send_json(
                        200,
                        app.select_skill(
                            data.get("learner", ""),
                            data.get("skill", ""),
                            request_id=req_id,
                        ),
                    )
                    return
                if path == "/api/ppo_step":
                    self._send_json(
                        200,
                        app.ppo_step(
                            data.get("learner", ""),
                            data.get("skill", ""),
                            action_id=data.get("action_id"),
                            policy_source=data.get("policy_source", "delayed_retention_ppo"),
                            format=data.get("format"),
                            difficulty=data.get("difficulty"),
                            question_source=data.get("question_source"),
                            mode=data.get("mode"),
                            request_id=req_id,
                        ),
                    )
                    return
                if path == "/api/hint":
                    self._send_json(
                        200,
                        app.hint(
                            data.get("learner", ""),
                            data.get("interaction", data.get("interaction_id", "")),
                            request_id=req_id,
                        ),
                    )
                    return
                if path == "/api/submit":
                    self._send_json(
                        200,
                        app.submit(
                            data.get("learner", ""),
                            data.get("interaction", data.get("interaction_id", "")),
                            data.get("answer"),
                            active_seconds=data.get("active_seconds", 5.0),
                            self_explanation=data.get("self_explanation"),
                            request_id=req_id,
                        ),
                    )
                    return
                if path == "/api/retry":
                    self._send_json(
                        200,
                        app.retry(
                            data.get("learner", ""),
                            data.get("interaction", data.get("interaction_id", "")),
                            request_id=req_id,
                        ),
                    )
                    return
                if path == "/api/reveal":
                    self._send_json(
                        200,
                        app.reveal(
                            data.get("learner", ""),
                            data.get("interaction", data.get("interaction_id", "")),
                            request_id=req_id,
                        ),
                    )
                    return
                if path == "/api/finish":
                    self._send_json(
                        200,
                        app.finish(
                            data.get("learner", ""),
                            data.get("interaction", data.get("interaction_id", "")),
                            request_id=req_id,
                        ),
                    )
                    return
            except (ValueError, KeyError, InvalidActionError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(500, {"error": f"Internal error: {exc}"})
                return

            self._send_json(404, {"error": f"Unknown route: {path}"})

        def _method_not_allowed(self) -> None:
            self._send_json(405, {"error": f"Method {self.command} not allowed."})

        do_PUT = _method_not_allowed
        do_DELETE = _method_not_allowed
        do_PATCH = _method_not_allowed

    return _AdaptiveHTTPServer(("127.0.0.1", int(port)), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the adaptive_web server.")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (default: 8780).")
    parser.add_argument("--db", type=str, default=None, help="SQLite database path.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (cpu, mps, cuda).")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    updates: dict[str, Any] = {}
    if args.port is not None:
        updates["adaptive_web_port"] = args.port
    if args.db is not None:
        updates["adaptive_web_db"] = Path(args.db)
    if args.device is not None:
        updates["device"] = args.device
    if updates:
        settings = replace(settings, **updates)

    app = AdaptiveLearningApp(settings)
    httpd = serve(app, port=settings.adaptive_web_port)
    print(f"adaptive_web listening on http://127.0.0.1:{httpd.server_port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
