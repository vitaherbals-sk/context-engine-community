# -*- coding: utf-8 -*-
"""End-to-end HTTP tests for non-ASCII payloads.

Regression cover for the report that writing a record with Slovak diacritics
over the SSE transport failed with an opaque
`{"logger": "mcp.server.exception_handler", "data": "Internal Server Error"}`
notification. These drive the real ASGI app over a real socket — uvicorn, the
Bearer middleware, the SSE transport, FastMCP argument validation and the
SQLite write — so an encoding regression anywhere in that chain fails here.
"""

import asyncio
import json
import os
import socket
import tempfile
import threading
import time

import pytest

# Full Slovak diacritic set — every character here is >1 byte in UTF-8.
SK_DIACRITICS = "áäčďéíľĺňóôŕšťúýž ÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ"
SK_TEXT = "výživové doplnky — " + SK_DIACRITICS

API_KEY = "test-api-key"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server():
    """Run the combined ASGI app on a real port, against a throwaway DB."""
    uvicorn = pytest.importorskip("uvicorn")

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    port = _free_port()
    keys = ("CTX_DB", "CTX_OAUTH_PASS", "CTX_API_KEY", "CTX_SERVER_URL", "CTX_HOST", "PORT")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update({
        "CTX_DB": db_path,
        "CTX_OAUTH_PASS": "test-oauth-pass",
        "CTX_API_KEY": API_KEY,
        "CTX_SERVER_URL": f"http://127.0.0.1:{port}",
        "CTX_HOST": "127.0.0.1",
        "PORT": str(port),
    })

    from context_engine import db as db_module
    from context_engine.combined_server import create_app

    # db.DB_PATH is captured at import time, which may have happened already.
    db_module.DB_PATH = db_path
    db_module.init_db(db_path)

    app, host, _ = create_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    if not server.started:
        raise RuntimeError("server did not start")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=10)
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        os.unlink(db_path)
    except OSError:
        pass


class _Session:
    """Minimal MCP SSE client — opens the stream, POSTs raw bytes, reads replies."""

    def __init__(self, client, base):
        self.client = client
        self.base = base
        self.messages = []
        self._endpoint = asyncio.get_running_loop().create_future()

    async def pump(self):
        headers = {"Authorization": f"Bearer {API_KEY}"}
        async with self.client.stream("GET", f"{self.base}/api/sse", headers=headers) as response:
            event = None
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
                    if event == "endpoint" and not self._endpoint.done():
                        self._endpoint.set_result(data)
                    else:
                        self.messages.append(json.loads(data))

    async def post(self, body: bytes):
        """POST exactly these bytes — no re-encoding, so tests control the wire format."""
        url = self.base + await asyncio.wait_for(self._endpoint, 10)
        return await self.client.post(url, content=body, headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        })

    async def send(self, payload: dict, ensure_ascii: bool = False):
        return await self.post(json.dumps(payload, ensure_ascii=ensure_ascii).encode("utf-8"))

    async def await_reply(self, msg_id: int, timeout: float = 10.0):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            for message in self.messages:
                if message.get("id") == msg_id:
                    return message
            await asyncio.sleep(0.05)
        raise AssertionError(f"no reply for id={msg_id}; received {self.messages}")

    def errors(self):
        return [m for m in self.messages
                if m.get("method") == "notifications/message"
                and m.get("params", {}).get("level") == "error"]


async def _with_session(base, scenario):
    httpx = pytest.importorskip("httpx")
    async with httpx.AsyncClient(timeout=30) as client:
        session = _Session(client, base)
        pump = asyncio.create_task(session.pump())
        try:
            await session.send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1"}}})
            await session.await_reply(1)
            await session.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            return await scenario(session)
        finally:
            pump.cancel()


def _run(base, scenario):
    return asyncio.run(_with_session(base, scenario))


def _result_text(reply):
    return reply["result"]["content"][0]["text"]


class TestDiacriticsRoundTrip:
    def test_raw_utf8_write_and_read_back(self, live_server):
        """Raw UTF-8 diacritics survive the POST, the SQLite write and the SSE reply."""
        company = "Vitaherbals " + SK_DIACRITICS

        async def scenario(session):
            await session.send({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "ctx_add_company",
                "arguments": {"name": company, "notes": SK_TEXT},
            }})
            created = await session.await_reply(2)

            await session.send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                "name": "ctx_company", "arguments": {"query": "Vitaherbals"},
            }})
            fetched = await session.await_reply(3)
            return created, fetched, session.errors()

        created, fetched, errors = _run(live_server, scenario)

        assert errors == [], f"server logged errors: {errors}"
        assert created["result"]["isError"] is False, created
        assert json.loads(_result_text(created))["status"] == "ok"

        text = _result_text(fetched)
        assert SK_TEXT in text, "diacritics were mangled on the way back"
        assert company in text

    def test_escaped_and_raw_bodies_are_equivalent(self, live_server):
        r"""A \uXXXX-escaped body and a raw UTF-8 body must give identical results."""
        query = "Vitaherbals " + SK_DIACRITICS

        async def scenario(session):
            await session.send({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
                "name": "ctx_find", "arguments": {"query": query},
            }}, ensure_ascii=True)
            escaped = await session.await_reply(4)

            await session.send({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
                "name": "ctx_find", "arguments": {"query": query},
            }}, ensure_ascii=False)
            raw = await session.await_reply(5)
            return escaped, raw, session.errors()

        escaped, raw, errors = _run(live_server, scenario)

        assert errors == [], f"server logged errors: {errors}"
        assert escaped["result"]["isError"] is False, escaped
        assert raw["result"]["isError"] is False, raw
        assert _result_text(escaped) == _result_text(raw)

    def test_long_diacritic_payload_is_not_truncated(self, live_server):
        """Byte length differs from character length — the body must arrive whole."""
        payload = SK_DIACRITICS * 200

        async def scenario(session):
            await session.send({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {
                "name": "ctx_add_note", "arguments": {
                    "title": "Dlhá poznámka s diakritikou",
                    "content": payload,
                    "domain": "work",
                    "category": "tech",
                    "tags": "test,diakritika",
                    "source": "pytest",
                },
            }})
            return await session.await_reply(6), session.errors()

        reply, errors = _run(live_server, scenario)

        assert errors == [], f"server logged errors: {errors}"
        assert reply["result"]["isError"] is False, reply


class TestMalformedBodyDiagnostics:
    """A broken body must say what is broken, not "Internal Server Error"."""

    def test_non_utf8_body_is_rejected_clearly(self, live_server):
        payload = {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {
            "name": "ctx_find", "arguments": {"query": SK_TEXT}}}

        async def scenario(session):
            # cp1250 is what a Windows console hands to curl for Slovak text.
            body = json.dumps(payload, ensure_ascii=False).encode("cp1250")
            response = await session.post(body)
            await asyncio.sleep(1.0)
            return response.status_code, response.text, session.errors()

        status, text, errors = _run(live_server, scenario)

        assert status == 400, text
        assert "not valid UTF-8" in text
        assert errors == [], "a malformed body must not surface as Internal Server Error"

    def test_truncated_body_is_rejected_clearly(self, live_server):
        payload = {"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {
            "name": "ctx_find", "arguments": {"query": SK_TEXT}}}

        async def scenario(session):
            text = json.dumps(payload, ensure_ascii=False)
            # Content-Length counted in characters, not bytes — the body arrives short.
            response = await session.post(text.encode("utf-8")[:len(text)])
            await asyncio.sleep(1.0)
            return response.status_code, response.text, session.errors()

        status, text, errors = _run(live_server, scenario)

        assert status == 400, text
        assert "not valid JSON" in text
        assert "Content-Length" in text
        assert errors == [], "a malformed body must not surface as Internal Server Error"
