"""The no-network invariant, enforced by a source scan in both languages.

WHY THIS IS A TEST AND NOT A CONVENTION

The studio this pack ships into runs with zero egress: a single box, three
engines, one model resident at a time, behind a local gateway, and nothing
leaves the machine. That is a property clients are told about, so it has to be
a property the build can prove -- not a habit that survives until the first
convenient `requests.get`.

It is also the single best reason a privacy-sensitive user installs this pack,
which means the day it quietly stops being true is the day the pack's main
selling point becomes a false claim. Treat this exactly like the widget-order
guard: an invariant with a test behind it.

WHAT COUNTS AS A VIOLATION

Outbound only. ComfyUI's own server is an aiohttp application and this pack
registers inbound routes on it (the Asset Bin asks the server to evaluate the
numbering rule rather than reimplementing it in JavaScript). Inbound route
registration is not egress, so `from aiohttp import web` is allowed and
`aiohttp.ClientSession` is not. The distinction is the whole point of the rule
and the scan encodes it rather than banning the package outright.

On the Python side the scan is AST-based, so a banned name inside a comment or
a docstring is not a false positive -- and, more importantly, a deferred
`import requests` in a function body still is one. On the JavaScript side there
is no parser available from here, so it is a text scan over the shipped `js/`
sources: absolute-origin URLs, the classic request constructors, and the
loaders that pull a CDN script or a webfont in.

WHAT IS DELIBERATELY NOT BANNED

URL *strings* in Python. Without an HTTP client in the package a URL literal
cannot do anything, and the fork's upstream repository address legitimately
appears in a module docstring. The import ban is the load-bearing half.

SCOPE

The shipped package only. `tests/` and `.github/` are excluded by design --
CI runners have network and are allowed to, and this file itself names every
banned symbol, so scanning the tests would fail on its own contents.
"""

import ast
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
JS_DIR = PROJECT_ROOT / "js"

# Not part of the shipped pack: the test suite and CI config are allowed network,
# the GPL reference copy is not ours, and the rest is build residue.
EXCLUDED_DIRS = {
    "tests", ".github", "__pycache__", ".git", ".venv", "venv",
    "reference_purpose only", "example workflow", "example_workflows",
}

# Outbound HTTP, sockets and mail/RPC transports. `http` covers http.client;
# `ssl` is here because a raw TLS socket is still egress.
BANNED_MODULES = {
    "requests", "httpx", "urllib", "urllib3", "socket", "ssl", "http",
    "ftplib", "smtplib", "poplib", "imaplib", "telnetlib", "xmlrpc",
    "webbrowser", "socketserver", "websockets", "websocket",
    # Model auto-download: the pack never fetches weights, it is handed a MODEL.
    "huggingface_hub", "torch.hub", "gdown", "wget",
}

# Allowed for the inbound Asset Bin routes; anything else from it is a client.
AIOHTTP_SERVER_NAMES = {"web"}

# Call names that download regardless of which module they were imported from.
BANNED_CALL_NAMES = {
    "urlopen", "urlretrieve", "hf_hub_download", "snapshot_download",
    "load_state_dict_from_url", "download_url_to_file", "sendBeacon",
}

# Attribute chains that reach a client through an otherwise-allowed package.
BANNED_ATTRIBUTES = {
    ("aiohttp", "ClientSession"), ("aiohttp", "request"),
    ("torch", "hub"), ("socket", "socket"),
}

# JavaScript request surfaces. `api.fetchApi` is ComfyUI's own same-origin
# client and is how the panel talks to this pack's routes -- it is not listed.
BANNED_JS_TOKENS = (
    "XMLHttpRequest",
    "sendBeacon",
    "EventSource",
    "importScripts",
    "new WebSocket",
    "@font-face",
    "@import",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "unpkg.com",
    "jsdelivr",
    "cdnjs",
)

# XML namespace identifiers. These look like URLs and are never dereferenced --
# `xmlns` on an inline SVG is a name, not an address. Nothing else is exempt.
ALLOWED_URIS = {
    "http://www.w3.org/2000/svg",
    "http://www.w3.org/1999/xlink",
    "http://www.w3.org/1999/xhtml",
}

URL_RE = re.compile(r"https?://[^\s\"'`)>]+")


def shipped_python_files():
    """Every .py file that ships, wherever the package directory is called.

    Discovered rather than listed so the scan survives the rename to
    comfyui_pulse_studio/ and so a new module cannot be added outside its reach.
    """
    found = []
    for path in PROJECT_ROOT.rglob("*.py"):
        rel = path.relative_to(PROJECT_ROOT)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        found.append(path)
    return sorted(found)


def shipped_js_files():
    return sorted(JS_DIR.glob("*.js")) if JS_DIR.is_dir() else []


def _dotted(node):
    """`a.b.c` from an attribute chain, or None if it is not a plain chain."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


class TestTheScanItselfWorks(unittest.TestCase):
    """A scan that silently finds no files passes forever. Prove it has input."""

    def test_python_sources_were_found(self):
        found = shipped_python_files()
        self.assertTrue(found, "the Python scan matched no files at all")
        names = {p.name for p in found}
        self.assertIn("nodes.py", names)
        self.assertIn("__init__.py", names)

    def test_javascript_sources_were_found(self):
        self.assertTrue(shipped_js_files(), "the JavaScript scan matched no files")

    def test_the_scan_excludes_the_test_suite(self):
        """This file names every banned symbol; scanning it would self-fail."""
        for path in shipped_python_files():
            self.assertNotEqual(path.name, Path(__file__).name,
                                "the scan is reading its own test file")

    def test_a_planted_violation_is_caught(self):
        """The detector, run against known-bad source rather than the tree."""
        bad = "def f():\n    import requests\n    return requests.get('x')\n"
        self.assertTrue(_import_violations(ast.parse(bad), Path("planted.py")),
                        "the AST walker missed a deferred `import requests`")


def _import_violations(tree, rel):
    """Banned imports in one parsed module, as human-readable strings."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base in BANNED_MODULES or alias.name in BANNED_MODULES:
                    out.append("%s:%d: import %s" % (rel, node.lineno, alias.name))
                elif base == "aiohttp":
                    # Bare `import aiohttp` exposes ClientSession; the routes do
                    # not need it and `from aiohttp import web` does not.
                    out.append("%s:%d: import aiohttp (use `from aiohttp import web`)"
                               % (rel, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue  # relative import: internal, never a transport
            base = node.module.split(".")[0]
            if base in BANNED_MODULES or node.module in BANNED_MODULES:
                out.append("%s:%d: from %s import ..." % (rel, node.lineno, node.module))
            elif base == "aiohttp":
                for alias in node.names:
                    if alias.name not in AIOHTTP_SERVER_NAMES:
                        out.append("%s:%d: from aiohttp import %s -- inbound routes "
                                   "only" % (rel, node.lineno, alias.name))
            for alias in node.names:
                if alias.name in BANNED_CALL_NAMES:
                    out.append("%s:%d: from %s import %s"
                               % (rel, node.lineno, node.module, alias.name))
    return out


class TestPythonMakesNoOutboundRequest(unittest.TestCase):
    def test_no_banned_import_reaches_the_shipped_package(self):
        violations = []
        for path in shipped_python_files():
            rel = path.relative_to(PROJECT_ROOT)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            violations.extend(_import_violations(tree, rel))
        self.assertEqual(violations, [],
                         "the pack must make no outbound request, ever:\n  "
                         + "\n  ".join(violations))

    def test_no_download_call_by_name(self):
        """Catches `urlopen(...)` reached through an alias the import scan allowed."""
        violations = []
        for path in shipped_python_files():
            rel = path.relative_to(PROJECT_ROOT)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name in BANNED_CALL_NAMES:
                    violations.append("%s:%d: %s(...)" % (rel, node.lineno, name))
        self.assertEqual(violations, [], "download call in the shipped package:\n  "
                                          + "\n  ".join(violations))

    def test_no_client_reached_through_an_allowed_package(self):
        violations = []
        banned = {"%s.%s" % pair for pair in BANNED_ATTRIBUTES}
        for path in shipped_python_files():
            rel = path.relative_to(PROJECT_ROOT)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                dotted = _dotted(node)
                if dotted in banned:
                    violations.append("%s:%d: %s" % (rel, node.lineno, dotted))
        self.assertEqual(violations, [], "client reached through an allowed import:\n  "
                                          + "\n  ".join(violations))

    def test_package_init_does_no_work_at_import_time(self):
        """ComfyUI imports every pack at startup. An update check here would be
        both egress and a visible startup regression for every user."""
        init = PROJECT_ROOT / "__init__.py"
        tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
        self.assertEqual(_import_violations(tree, init.name), [])


class TestJavaScriptMakesNoOutboundRequest(unittest.TestCase):
    def test_no_absolute_origin_url_in_the_widget_layer(self):
        violations = []
        for path in shipped_js_files():
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for url in URL_RE.findall(line):
                    if url.rstrip("\"'`") in ALLOWED_URIS:
                        continue
                    violations.append("%s:%d: %s" % (path.name, lineno, url))
        self.assertEqual(violations, [],
                         "absolute-origin URL in the widget layer. Fonts come from "
                         "the ComfyUI stylesheet and icons are inline SVG:\n  "
                         + "\n  ".join(violations))

    def test_no_request_constructor_or_cdn_loader(self):
        violations = []
        for path in shipped_js_files():
            source = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(source.splitlines(), 1):
                for token in BANNED_JS_TOKENS:
                    if token in line:
                        violations.append("%s:%d: %s" % (path.name, lineno, token))
        self.assertEqual(violations, [], "banned request surface in the widget layer:\n  "
                                          + "\n  ".join(violations))

    def test_the_allowed_uris_are_only_ever_xml_namespaces(self):
        """The exemption is narrow on purpose: an allowed URI outside an `xmlns`
        position would be a real fetch wearing a namespace's clothes."""
        for path in shipped_js_files():
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for uri in ALLOWED_URIS:
                    if uri in line:
                        self.assertIn("xmlns", line,
                                      "%s:%d: %s is exempt only as an xmlns value"
                                      % (path.name, lineno, uri))


if __name__ == "__main__":
    unittest.main()
