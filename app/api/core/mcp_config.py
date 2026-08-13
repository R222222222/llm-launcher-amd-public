"""Canonical, local MCP stdio configuration contract."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .constants import MCP_CONFIG_FILE

_ALLOWED_SERVER_FIELDS = frozenset({"command", "args", "env", "cwd", "timeout_ms"})


class McpConfigError(ValueError):
    """A safe-to-display MCP path or schema error (never includes values)."""


def _no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise McpConfigError("MCP servers config contém symlink")


def validate_path(value: str | Path) -> Path:
    """Require exactly the absolute canonical real config path."""
    if not isinstance(value, (str, Path)):
        raise McpConfigError("MCP servers config inválido")
    path = Path(value)
    canonical = MCP_CONFIG_FILE
    if not path.is_absolute() or path != canonical:
        raise McpConfigError("MCP servers config precisa ser config/mcp/servers.json")
    if path.suffix.lower() != ".json":
        raise McpConfigError("MCP servers config precisa ser JSON")
    try:
        _no_symlink_components(path)
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise McpConfigError("MCP servers config não existe") from exc
    except OSError as exc:
        raise McpConfigError("MCP servers config não pôde ser resolvido") from exc
    if resolved != canonical or not path.is_file() or not os.path.isfile(path):
        raise McpConfigError("MCP servers config não é arquivo regular")
    return path


def _schema_error(message: str) -> McpConfigError:
    return McpConfigError(message)


def validate(value: str | Path) -> tuple[Path, dict[str, Any]]:
    path = validate_path(value)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _schema_error("MCP servers config não é JSON válido") from exc
    if not isinstance(data, dict):
        raise _schema_error("MCP servers config deve ser um objeto")
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        raise _schema_error("mcpServers deve ser um objeto")
    for name, entry in servers.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise _schema_error("entrada MCP inválida")
        if set(entry) - _ALLOWED_SERVER_FIELDS:
            raise _schema_error("entrada MCP contém campo não permitido")
        command = entry.get("command")
        if not isinstance(command, str) or not command or not Path(command).is_absolute():
            raise _schema_error("command MCP deve ser um caminho absoluto")
        if "args" in entry and (
            not isinstance(entry["args"], list)
            or any(not isinstance(arg, str) for arg in entry["args"])
        ):
            raise _schema_error("args MCP deve ser uma lista de strings")
        if "env" in entry and (
            not isinstance(entry["env"], dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in entry["env"].items())
        ):
            raise _schema_error("env MCP deve ser um objeto de strings")
        if "cwd" in entry and (
            not isinstance(entry["cwd"], str)
            or not entry["cwd"]
            or not Path(entry["cwd"]).is_absolute()
        ):
            raise _schema_error("cwd MCP deve ser um caminho absoluto")
        if "timeout_ms" in entry and (
            isinstance(entry["timeout_ms"], bool)
            or not isinstance(entry["timeout_ms"], int)
            or entry["timeout_ms"] <= 0
        ):
            raise _schema_error("timeout_ms MCP deve ser positivo")
    return path, data
