"""Combined server — OAuth + API key MCP SSE behind one port for Railway.

Two access methods:
1. /sse — OAuth 2.1 (for Cowork, supports DCR)
2. /api/sse — API key Bearer token (for OpenClaw and other clients)

Uses two FastMCP instances sharing the same DB — one with OAuth, one without.
Tools are registered via shared register_tools() function.
"""

import os
import sys
import json
import asyncio

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from uvicorn import Config, Server

from mcp_oauth import AuthServerSettings, SimpleAuthSettings
from mcp_oauth.server.auth_provider.simple_auth_provider import SimpleOAuthProvider
from mcp_oauth.server.features.functions import ExtraFunctions
from mcp.server.auth.routes import create_auth_routes
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.transport_security import TransportSecuritySettings

TOKEN_LIFETIME = 365 * 24 * 3600


# Max body we buffer for validation — mirrors SseServerTransport's own 4 MiB
# limit, so anything bigger is rejected downstream anyway.
MAX_VALIDATED_BODY = 4 * 1024 * 1024


class JSONBodyGuardMiddleware:
    """ASGI middleware — reject malformed POST bodies with a readable error.

    The MCP SSE transport answers an unparseable body with a bare
    400 "Could not parse message" and pushes the ValidationError into the
    session stream, where it surfaces as an opaque
    {"logger": "mcp.server.exception_handler", "data": "Internal Server Error"}
    notification. That hides the two things that actually go wrong in practice:

    1. The body is not valid UTF-8 (a shell re-encoded the payload into the
       console codepage, e.g. cp1250, so "vyzivove" with diacritics arrives as
       raw high bytes).
    2. The body is valid UTF-8 but truncated, because Content-Length counted
       characters instead of bytes — non-ASCII text is >1 byte per character.

    Both are indistinguishable from a server bug unless we say so. This
    middleware decodes and json-parses the body first and returns a concrete
    diagnosis; well-formed bodies are replayed downstream untouched.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        chunks = []
        size = 0
        oversized = False
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                chunks.append(message)
                break
            chunks.append(message)
            size += len(message.get("body", b""))
            more_body = message.get("more_body", False)
            if size > MAX_VALIDATED_BODY:
                oversized = True
                break

        replay = _replay_receive(chunks, receive, exhausted=not more_body)

        if not oversized:
            body = b"".join(m.get("body", b"") for m in chunks if m["type"] == "http.request")
            error = _diagnose_body(body)
            if error:
                response = JSONResponse(
                    {"error": "invalid_request_body", "detail": error},
                    status_code=400,
                )
                await response(scope, receive, send)
                return

        await self.app(scope, replay, send)


def _replay_receive(chunks, receive: Receive, exhausted: bool) -> Receive:
    """Hand the buffered messages back to the downstream app, in order."""
    pending = list(chunks)

    async def _receive():
        if pending:
            return pending.pop(0)
        if exhausted:
            return {"type": "http.request", "body": b"", "more_body": False}
        return await receive()

    return _receive


def _diagnose_body(body: bytes) -> str | None:
    """Return a human-readable reason the body is unusable, or None if it's fine."""
    if not body:
        return None  # let the transport handle empty bodies as before

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as err:
        snippet = body[max(0, err.start - 20):err.start + 20]
        return (
            f"Request body is not valid UTF-8: {err.reason} at byte {err.start} "
            f"(offending bytes: {body[err.start:err.end]!r}, context: {snippet!r}). "
            "The body was most likely re-encoded by the shell into a legacy "
            "codepage. Send the JSON as UTF-8 bytes (e.g. curl --data-binary @file.json "
            "with a UTF-8 file), or escape non-ASCII using JSON unicode escapes."
        )

    try:
        json.loads(text)
    except json.JSONDecodeError as err:
        hint = ""
        if len(body) != len(text):
            hint = (
                f" The body is {len(body)} bytes but {len(text)} characters — if the "
                "client set Content-Length from the character count, the body arrived "
                "truncated. Content-Length must be the UTF-8 byte length."
            )
        return f"Request body is not valid JSON: {err.msg} (line {err.lineno}, column {err.colno}).{hint}"

    return None


class BearerTokenMiddleware:
    """ASGI middleware — checks Bearer token or ?token= query param."""
    def __init__(self, app: ASGIApp, api_key: str):
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode()

            query_string = scope.get("query_string", b"").decode()
            token_param = ""
            for param in query_string.split("&"):
                if param.startswith("token="):
                    token_param = param[6:]
                    break

            if auth_header == f"Bearer {self.api_key}" or token_param == self.api_key:
                await self.app(scope, receive, send)
                return

            response = PlainTextResponse(
                "Unauthorized. Use: Authorization: Bearer <api_key>", status_code=401)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def create_app():
    port = int(os.environ.get("PORT", os.environ.get("CTX_PORT", "8000")))
    host = os.environ.get("CTX_HOST", "0.0.0.0")

    oauth_pass = os.environ.get("CTX_OAUTH_PASS")
    oauth_user = os.environ.get("CTX_OAUTH_USER", "satori")
    server_url = os.environ.get("CTX_SERVER_URL",
                                "https://unique-healing-production-7a14.up.railway.app")
    api_key = os.environ.get("CTX_API_KEY", oauth_pass)

    if not oauth_pass:
        print("ERROR: CTX_OAUTH_PASS required", file=sys.stderr)
        sys.exit(1)

    # ── OAuth routes ──────────────────────────────────────────
    auth_settings = SimpleAuthSettings(
        superusername=oauth_user,
        superuserpassword=oauth_pass,
        mcp_scope="user",
    )
    oauth_provider = SimpleOAuthProvider(
        settings=auth_settings,
        auth_callback_url=f"{server_url}/login",
        server_url=server_url,
        expired_at=TOKEN_LIFETIME,
    )
    mcp_auth_settings = AuthSettings(
        issuer_url=server_url,
        client_registration_options=ClientRegistrationOptions(
            enabled=True, valid_scopes=["user"], default_scopes=["user"],
        ),
        required_scopes=["user"],
        resource_server_url=None,
    )
    oauth_routes = create_auth_routes(
        provider=oauth_provider,
        issuer_url=mcp_auth_settings.issuer_url,
        service_documentation_url=mcp_auth_settings.service_documentation_url,
        client_registration_options=mcp_auth_settings.client_registration_options,
        revocation_options=mcp_auth_settings.revocation_options,
    )
    ExtraFunctions(oauth_provider=oauth_provider).append_functions(oauth_routes)

    # ── MCP instance 1: OAuth-protected (Cowork) ─────────────
    os.environ["CTX_OAUTH_URL"] = server_url
    os.environ["CTX_SERVER_URL"] = server_url

    from context_engine.server import mcp  # This picks up CTX_OAUTH_URL → creates OAuth MCP
    mcp.settings.port = port
    mcp.settings.host = host
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    )
    oauth_mcp_app = JSONBodyGuardMiddleware(mcp.sse_app())

    # ── MCP instance 2: No auth (API key handled by middleware) ──
    from mcp.server.fastmcp import FastMCP

    mcp_api = FastMCP(
        "Context Engine",
        instructions="Strukturovana kontextova pamat — ludia, firmy, projekty, pravidla, poznamky. Life OS.",
    )
    mcp_api.settings.port = port
    mcp_api.settings.host = host
    mcp_api.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    )
    # Copy all tools from OAuth instance to API key instance
    for name, tool in mcp._tool_manager._tools.items():
        mcp_api._tool_manager._tools[name] = tool
    api_mcp_app = BearerTokenMiddleware(JSONBodyGuardMiddleware(mcp_api.sse_app()), api_key)

    # ── Utility endpoints ─────────────────────────────────────
    async def health(request):
        return PlainTextResponse("ok")

    # RFC 9728 protected resource metadata — without it a client cannot discover
    # how this server signs in, and OAuth setup has to be configured by hand.
    # Served at both the bare path and the /sse suffix clients derive from the
    # resource URL.
    issuer = str(mcp_auth_settings.issuer_url)

    async def protected_resource_metadata(request):
        return JSONResponse({
            "resource": f"{server_url}/sse",
            "authorization_servers": [issuer],
            "scopes_supported": ["user"],
            "bearer_methods_supported": ["header"],
        })

    async def upload_db(request):
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {oauth_pass}":
            return PlainTextResponse("unauthorized", status_code=401)
        body = await request.body()
        if not body:
            return PlainTextResponse("empty body", status_code=400)
        db_path = os.environ.get("CTX_DB", "/data/context-engine.db")
        with open(db_path, "wb") as f:
            f.write(body)
        return PlainTextResponse(f"ok, wrote {len(body)} bytes to {db_path}")

    # ── Routes ────────────────────────────────────────────────
    all_routes = [
        Route("/health", health),
        Route("/.well-known/oauth-protected-resource", protected_resource_metadata),
        Route("/.well-known/oauth-protected-resource/sse", protected_resource_metadata),
        Route("/admin/upload-db", upload_db, methods=["POST"]),
        Mount("/api", app=api_mcp_app),   # API key: /api/sse, /api/messages/
    ] + oauth_routes + [
        Mount("/", app=oauth_mcp_app),    # OAuth: /sse, /messages/
    ]

    return Starlette(routes=all_routes), host, port


def main():
    app, host, port = create_app()

    async def run():
        config = Config(app, host=host, port=port, log_level="info")
        server = Server(config)
        await server.serve()

    from context_engine.db import init_db
    init_db()

    print(f"Combined OAuth + API key MCP server on {host}:{port}")
    asyncio.run(run())


if __name__ == "__main__":
    main()
