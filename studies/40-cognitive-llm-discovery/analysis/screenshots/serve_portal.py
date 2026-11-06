"""Shinka's portal for a running search, on the search's node: read-only, on 127.0.0.1 only.

Usage: python serve_portal.py <root with search/programs.sqlite> -p <port>
"""
import socketserver
import sqlite3
import sys

_connect = sqlite3.connect


def _read_only(path, *a, **k):
    if isinstance(path, str) and path.startswith("file:"):      # Shinka's own read-only URI
        return _connect(path, *a, **k)
    k.pop("uri", None)
    return _connect(f"file:{path}?mode=ro", *a, uri=True, **k)


sqlite3.connect = _read_only
_init = socketserver.TCPServer.__init__
socketserver.TCPServer.__init__ = lambda self, addr, *a, **k: _init(self, ("127.0.0.1", addr[1]), *a, **k)

from shinka.webui.visualization import main  # noqa: E402

sys.argv[0] = "shinka_visualize"
main()
