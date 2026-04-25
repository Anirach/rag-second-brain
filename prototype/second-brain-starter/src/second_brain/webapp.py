from __future__ import annotations

import cgi
import json
import tempfile
from dataclasses import asdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .answer import answer_query, format_answer
from .config import load_settings
from .dashboard import dashboard_payload
from .ingest import ingest_file
from .query import format_results, search
from .redundancy import merge_objects, scan_redundancy
from .review import list_reviews, resolve_review
from .status import get_object_status, list_object_statuses, promote_object_status, set_object_status


def run_server(root: str | Path = '.', host: str = '127.0.0.1', port: int = 8765) -> None:
    root_path = Path(root).resolve()
    settings = load_settings(root_path)
    static_dir = root_path / 'web'

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_dir), **kwargs)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith('/api/'):
                return self._handle_api(parsed)
            if parsed.path == '/':
                self.path = '/index.html'
            return super().do_GET()

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == '/api/upload':
                return self._handle_upload()
            if parsed.path == '/api/reviews/resolve':
                return self._handle_review_resolve()
            if parsed.path == '/api/objects/status':
                return self._handle_object_status()
            if parsed.path == '/api/redundancy/merge':
                return self._handle_merge()
            self.send_error(HTTPStatus.NOT_FOUND, 'Unknown API route')

        def log_message(self, format, *args):
            return

        def _handle_api(self, parsed):
            params = parse_qs(parsed.query)
            try:
                if parsed.path == '/api/query':
                    q = params.get('q', [''])[0]
                    payload = {'text': format_results(search(settings, q))}
                elif parsed.path == '/api/answer':
                    q = params.get('q', [''])[0]
                    payload = {'text': format_answer(answer_query(settings, q))}
                elif parsed.path == '/api/objects':
                    status = params.get('status', [''])[0] or None
                    payload = {'items': [asdict(item) for item in list_object_statuses(settings, status=status)]}
                elif parsed.path == '/api/reviews':
                    status = params.get('status', ['open'])[0]
                    payload = {'items': [asdict(item) for item in list_reviews(settings, status=status)]}
                elif parsed.path == '/api/dashboard':
                    payload = dashboard_payload(settings)
                elif parsed.path == '/api/redundancy':
                    payload = {'items': [asdict(item) for item in scan_redundancy(settings)]}
                elif parsed.path == '/api/artifact':
                    object_id = params.get('object_id', [''])[0]
                    if not object_id:
                        self.send_error(HTTPStatus.BAD_REQUEST, 'Missing object_id')
                        return
                    item = get_object_status(settings, object_id)
                    if not item:
                        self.send_error(HTTPStatus.NOT_FOUND, 'Object not found')
                        return
                    path = settings.root / item.path
                    payload = {'item': asdict(item), 'content': path.read_text(encoding='utf-8')}
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, 'Unknown API route')
                    return
            except Exception as exc:
                return self._text_error(exc)

            return self._json_response(payload)

        def _handle_upload(self):
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        'REQUEST_METHOD': 'POST',
                        'CONTENT_TYPE': self.headers.get('Content-Type', ''),
                    },
                )
                file_item = form['file'] if 'file' in form else None
                if file_item is None or not getattr(file_item, 'filename', None):
                    self.send_error(HTTPStatus.BAD_REQUEST, 'Missing file field')
                    return
                title = form.getfirst('title') or None
                origin = form.getfirst('origin') or 'web-ui'

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir) / Path(file_item.filename).name
                    data = file_item.file.read()
                    tmp_path.write_bytes(data)
                    result = ingest_file(settings, tmp_path, title=title, origin=origin)

                payload = {
                    'ok': True,
                    'source_id': result.source_id,
                    'chunk_ids': result.chunk_ids,
                    'entity_ids': result.entity_ids,
                    'concept_id': result.concept_id,
                    'synthesis_id': result.synthesis_id,
                }
                return self._json_response(payload)
            except ValueError as exc:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(str(exc).encode('utf-8'))
            except Exception as exc:
                return self._text_error(exc)

        def _handle_review_resolve(self):
            try:
                payload = self._read_json_body()
                review_id = payload.get('review_id')
                decision = payload.get('decision')
                notes = payload.get('notes', '')
                promote = bool(payload.get('promote', False))
                promote_target = payload.get('promote_target')
                if not review_id or not decision:
                    self.send_error(HTTPStatus.BAD_REQUEST, 'Missing review_id or decision')
                    return
                item = resolve_review(settings, review_id, decision, notes=notes, promote=promote, promote_target=promote_target)
                if not item:
                    self.send_error(HTTPStatus.NOT_FOUND, 'Review not found')
                    return
                return self._json_response({'ok': True, 'item': asdict(item)})
            except ValueError as exc:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(str(exc).encode('utf-8'))
            except Exception as exc:
                return self._text_error(exc)

        def _handle_object_status(self):
            try:
                payload = self._read_json_body()
                object_id = payload.get('object_id')
                action = payload.get('action', 'set')
                status = payload.get('status')
                reason = payload.get('reason', 'updated via web-ui')
                if not object_id:
                    self.send_error(HTTPStatus.BAD_REQUEST, 'Missing object_id')
                    return
                if action == 'promote':
                    item = promote_object_status(settings, object_id, target=status, reason=reason)
                else:
                    if not status:
                        self.send_error(HTTPStatus.BAD_REQUEST, 'Missing status')
                        return
                    item = set_object_status(settings, object_id, status, reason=reason)
                if not item:
                    self.send_error(HTTPStatus.NOT_FOUND, 'Object not found')
                    return
                return self._json_response({'ok': True, 'item': asdict(item)})
            except ValueError as exc:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(str(exc).encode('utf-8'))
            except Exception as exc:
                return self._text_error(exc)

        def _handle_merge(self):
            try:
                payload = self._read_json_body()
                source_object_id = payload.get('source_object_id')
                target_object_id = payload.get('target_object_id')
                reason = payload.get('reason', 'merged via web-ui')
                if not source_object_id or not target_object_id:
                    self.send_error(HTTPStatus.BAD_REQUEST, 'Missing source_object_id or target_object_id')
                    return
                item = merge_objects(settings, source_object_id, target_object_id, reason=reason)
                return self._json_response({'ok': True, 'item': asdict(item)})
            except ValueError as exc:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(str(exc).encode('utf-8'))
            except Exception as exc:
                return self._text_error(exc)

        def _read_json_body(self):
            length = int(self.headers.get('Content-Length', '0'))
            raw = self.rfile.read(length) if length > 0 else b'{}'
            return json.loads(raw.decode('utf-8') or '{}')

        def _json_response(self, payload: dict):
            body = json.dumps(payload).encode('utf-8')
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _text_error(self, exc: Exception):
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(str(exc).encode('utf-8'))

    server = ThreadingHTTPServer((host, port), Handler)
    print(f'Second Brain UI running at http://{host}:{port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
