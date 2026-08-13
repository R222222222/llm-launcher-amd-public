"""Static Phase 6 checklist and evidence gate.

The checklist is deliberately data-only: scenarios can be added later without
changing the runner's lifecycle or its mutation guard.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUSES = ("PASS", "FAIL", "NÃO VERIFICADO")


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    section: str
    spec_line: int
    action: str
    expected: str
    critical: bool = False


def _items() -> tuple[ChecklistItem, ...]:
    """Every acceptance bullet in SPEC_FINAL Fase 6, with stable IDs."""
    rows: list[ChecklistItem] = [
        ChecklistItem("CP-01", "caminho crítico", 247, "baseline", "three baseline VRAM readings are captured", True),
        ChecklistItem("CP-02", "caminho crítico", 248, "config", "the allowlisted small-model config is persisted", True),
        ChecklistItem("CP-03", "caminho crítico", 248, "subir perfil", "UI launches the allowlisted small model", True),
        ChecklistItem("CP-04", "caminho crítico", 248, "health", "8421 /health returns 200/status ok", True),
        ChecklistItem("CP-05", "caminho crítico", 248, "modal", "modal shows attempt and command", True),
        ChecklistItem("CP-06", "caminho crítico", 248, "completion", "200, non-empty choices/content and finish_reason", True),
        ChecklistItem("CP-07", "caminho crítico", 248, "Stop", "Stop is clicked through the UI", True),
        ChecklistItem("CP-08", "caminho crítico", 248, "higiene", "launches and registry are empty; 8421 is closed", True),
        ChecklistItem("CP-09", "caminho crítico", 248, "VRAM", "three stable post-stop readings within baseline +64 MiB", True),
        ChecklistItem("CP-10", "caminho crítico", 249, "telemetria", "API GPU agrees with sysfs bracket within 1 MiB", True),
        ChecklistItem("CONFIG-01", "Configs", 253, "filtro", "filter by alias/backend/path works", False),
        ChecklistItem("CONFIG-02", "Configs", 253, "backend chips", "backend chips filter rows", False),
        ChecklistItem("CONFIG-03", "Configs", 253, "estimativa", "estimate status dot is visible", False),
        ChecklistItem("CONFIG-04", "Configs", 253, "editar", "edit opens the selected row", False),
        ChecklistItem("CONFIG-05", "Configs", 254, "duplicar", "duplicate preserves fields and creates an ID", False),
        ChecklistItem("CONFIG-06", "Configs", 254, "excluir", "delete removes a config", False),
        ChecklistItem("CONFIG-07", "Configs", 254, "+ nova", "new config editor opens", False),
        ChecklistItem("CONFIG-08", "Configs", 254, "launch", "single launch is available", False),
        ChecklistItem("CONFIG-09", "Configs", 254, "router", "multi-select router launch is available", False),
        ChecklistItem("CONFIG-10", "Configs", 254, "Stop", "active launch can be stopped", False),
        ChecklistItem("CONFIG-11", "Configs", 255, "Restart/logs/409", "restart, logs and second-launch 409 are covered", False),
        ChecklistItem("EDITOR-01", "Editor", 257, "modelo", "model dropdown works", False),
        ChecklistItem("EDITOR-02", "Editor", 257, "llama.cpp decide", "auto mode disables and restores tuning fields", False),
        ChecklistItem("EDITOR-03", "Editor", 258, "backend", "unavailable backend is disabled", False),
        ChecklistItem("EDITOR-04", "Editor", 258, "ctx/slots/KV", "context, slots and valid KV choices work", False),
        ChecklistItem("EDITOR-05", "Editor", 259, "flash/ngl/ncmoe", "flash, GPU layers and MoE controls work", False),
        ChecklistItem("EDITOR-06", "Editor", 259, "cache", "cache-ram and ctx-checkpoints work", False),
        ChecklistItem("EDITOR-07", "Editor", 260, "generation", "max tokens, batch, ubatch and threads work", False),
        ChecklistItem("EDITOR-08", "Editor", 260, "flags", "mlock and verbose work", False),
        ChecklistItem("EDITOR-09", "Editor", 261, "server/cli", "server/cli selector works", False),
        ChecklistItem("EDITOR-10", "Editor", 261, "panels", "live estimate and command panels work", False),
        ChecklistItem("EDITOR-11", "Editor", 262, "save/launch", "Save and Save+launch work", False),
        ChecklistItem("EDITOR-12", "Editor", 263, "single GPU", "split mode absence is recorded as hardware limit", False),
        ChecklistItem("LOG-01", "Modal de logs", 264, "resumo/comando", "summary and command are shown", False),
        ChecklistItem("LOG-02", "Modal de logs", 264, "SSE", "stdout SSE stream is shown", False),
        ChecklistItem("LOG-03", "Modal de logs", 264, "tentativas", "attempts are numbered", False),
        ChecklistItem("LOG-04", "Modal de logs", 265, "reiniciar/cancelar", "restart and cancel buttons work", False),
        ChecklistItem("LOG-05", "Modal de logs", 265, "esconder/reabrir", "modal can hide and reopen", False),
        ChecklistItem("MODELS-01", "Models", 266, "listagem", "models are listed", False),
        ChecklistItem("MODELS-02", "Models", 266, "shards", "shard size is summed", False),
        ChecklistItem("MODELS-03", "Models", 266, "flags/filtro", "flags and filter work", False),
        ChecklistItem("MODELS-04", "Models", 267, "delete", "only the exact E2E model can be deleted", False),
        ChecklistItem("DOWNLOAD-01", "Download", 269, "destino", "download destination is selectable", False),
        ChecklistItem("DOWNLOAD-02", "Download", 269, "URL/owner", "URL and owner/repo inspection work", False),
        ChecklistItem("DOWNLOAD-03", "Download", 269, "busca", "search works", False),
        ChecklistItem("DOWNLOAD-04", "Download", 269, "quantização", "quantization grouping works", False),
        ChecklistItem("DOWNLOAD-05", "Download", 270, "progresso", "progress and speed are shown", False),
        ChecklistItem("DOWNLOAD-06", "Download", 270, "cancelar", "download can be cancelled", False),
        ChecklistItem("DOWNLOAD-07", "Download", 270, "troca de aba", "download survives tab changes", False),
        ChecklistItem("DOWNLOAD-08", "Download", 270, "integridade", "integrity is validated", False),
        ChecklistItem("DOWNLOAD-09", "Download", 270, "sampling", "sampling.json is generated", False),
        ChecklistItem("MCP-01", "MCP", 272, "loopback", "MCP tab is enabled only on loopback", False),
        ChecklistItem("MCP-02", "MCP", 272, "cadastrar/ligar", "server can be registered and started", False),
        ChecklistItem("MCP-03", "MCP", 273, "desligar", "server can be stopped", False),
        ChecklistItem("MCP-04", "MCP", 273, "logs/editar/excluir", "logs, edit and delete work", False),
        ChecklistItem("MCP-05", "MCP", 273, "auto-stop", "non-zero server exits auto-stop", False),
        ChecklistItem("MCP-06", "MCP", 273, "404", "MCP is disabled and 404 after cleanup", False),
        ChecklistItem("GPU-01", "AMD GPU", 274, "refresh", "auto-refresh and refresh button work", False),
        ChecklistItem("GPU-02", "AMD GPU", 275, "VRAM", "total, used and free VRAM are shown", False),
        ChecklistItem("GPU-03", "AMD GPU", 275, "sensores", "temperature, utilization and fan are shown", False),
        ChecklistItem("GPU-04", "AMD GPU", 275, "power/clocks", "power and clocks are shown", False),
        ChecklistItem("GPU-05", "AMD GPU", 276, "histórico", "VRAM/RAM history charts are shown", False),
        ChecklistItem("GPU-06", "AMD GPU", 276, "cards", "one card per GPU is shown", False),
        ChecklistItem("GPU-07", "AMD GPU", 276, "sysfs", "telemetry agrees with sysfs", False),
        ChecklistItem("HEADER-01", "Header", 277, "VRAM/RAM", "header bars are shown", False),
        ChecklistItem("HEADER-02", "Header", 277, "backend", "backend badges are shown", False),
        ChecklistItem("HEADER-03", "Header", 277, "refresh", "header refresh works", False),
        ChecklistItem("SETTINGS-01", "Settings", 278, "model paths", "model folders can be configured", False),
        ChecklistItem("SETTINGS-02", "Settings", 278, "backend dirs", "backend directories can be configured", False),
        ChecklistItem("SETTINGS-03", "Settings", 279, "default", "default directory is shown", False),
        ChecklistItem("SETTINGS-04", "Settings", 279, "reset", "reset button works", False),
        ChecklistItem("SETTINGS-05", "Settings", 279, "custom", "custom without path is not launchable", False),
        ChecklistItem("SETTINGS-06", "Settings", 280, "remote 403", "remote settings POST returns 403", False),
    ]
    return tuple(rows)


CHECKLIST_ITEMS = _items()
CHECKLIST_BY_ID = {item.id: item for item in CHECKLIST_ITEMS}


@dataclass
class ChecklistResult:
    status: str = "NÃO VERIFICADO"
    action: str = ""
    expected: str = ""
    observed: str = ""
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    timestamp: str = ""
    duration_seconds: float | None = None


class Checklist:
    """Mutable run report with a strict PASS/evidence and completeness gate."""

    def __init__(self, run_id: str, *, suite_complete: bool = False):
        self.run_id = run_id
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.suite_complete = suite_complete
        self.results = {item.id: ChecklistResult() for item in CHECKLIST_ITEMS}

    @staticmethod
    def validate_manifest() -> None:
        ids = [item.id for item in CHECKLIST_ITEMS]
        if len(ids) != len(set(ids)) or not ids:
            raise RuntimeError("checklist sem IDs estáveis únicos")
        if any(item.spec_line <= 0 or not item.action or not item.expected for item in CHECKLIST_ITEMS):
            raise RuntimeError("checklist incompleto")
        critical = [item.id for item in CHECKLIST_ITEMS if item.critical]
        if not critical:
            raise RuntimeError("checklist sem caminho crítico")

    def record(
        self,
        item_id: str,
        status: str,
        *,
        observed: str,
        evidence: list[str] | tuple[str, ...] = (),
        reason: str = "",
        duration_seconds: float | None = None,
    ) -> None:
        item = CHECKLIST_BY_ID.get(item_id)
        if item is None:
            raise KeyError(f"item desconhecido: {item_id}")
        if status not in STATUSES:
            raise ValueError(f"status inválido: {status}")
        evidence_list = [str(path) for path in evidence if str(path)]
        if status == "PASS" and (not observed or not evidence_list):
            raise ValueError(f"PASS sem evidência: {item_id}")
        if status == "FAIL" and not evidence_list:
            raise ValueError(f"FAIL sem evidência: {item_id}")
        self.results[item_id] = ChecklistResult(
            status=status,
            action=item.action,
            expected=item.expected,
            observed=observed,
            reason=reason,
            evidence=evidence_list,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration_seconds,
        )

    def mark_dependents_unverified(self, reason: str) -> None:
        for item in CHECKLIST_ITEMS:
            if self.results[item.id].status == "NÃO VERIFICADO":
                self.record(item.id, "NÃO VERIFICADO", observed="", reason=reason)

    def fail_critical(self, item_id: str, reason: str, evidence: list[str]) -> None:
        self.record(item_id, "FAIL", observed="runner failure", reason=reason, evidence=evidence)
        self.mark_dependents_unverified(reason)

    def mark_unimplemented(self, reason: str = "cenário ainda não implementado nesta fundação") -> None:
        for item in CHECKLIST_ITEMS:
            if (
                not item.critical
                and self.results[item.id].status == "NÃO VERIFICADO"
                and not self.results[item.id].reason
            ):
                self.record(item.id, "NÃO VERIFICADO", observed="", reason=reason)

    def gate_pass(self) -> bool:
        return self.suite_complete and all(
            self.results[item.id].status == "PASS" for item in CHECKLIST_ITEMS if item.critical
        )

    def validate_report(self, evidence_dir: Path | None = None) -> None:
        self.validate_manifest()
        if set(self.results) != set(CHECKLIST_BY_ID):
            raise RuntimeError("relatório não cobre todos os itens do manifesto")
        for item in CHECKLIST_ITEMS:
            result = self.results[item.id]
            if result.status not in STATUSES:
                raise RuntimeError(f"status inválido no relatório: {item.id}")
            if result.status == "PASS" and (not result.observed or not result.evidence):
                raise RuntimeError(f"PASS sem evidência no relatório: {item.id}")
            if result.status in {"FAIL", "NÃO VERIFICADO"} and not result.reason:
                raise RuntimeError(f"{result.status} sem motivo no relatório: {item.id}")
            if result.status == "FAIL" and not result.evidence:
                raise RuntimeError(f"FAIL sem evidência no relatório: {item.id}")
            if evidence_dir is not None:
                file_references = 0
                for reference in result.evidence:
                    # Human-readable endpoint references are useful evidence
                    # labels, but they are not files and must not satisfy the
                    # final artifact existence gate.
                    if reference.startswith(("API ", "HTTP ", "GET ", "POST ", "DELETE ")):
                        continue
                    candidate = Path(reference)
                    if candidate.is_absolute() or not (evidence_dir / candidate).is_file():
                        raise RuntimeError(f"evidência ausente ou absoluta: {item.id}: {reference}")
                    file_references += 1
                if result.status in {"PASS", "FAIL"} and file_references == 0:
                    raise RuntimeError(f"{result.status} exige ao menos uma evidência de arquivo: {item.id}")

    def as_dict(self, evidence_dir: Path | None = None) -> dict[str, Any]:
        self.validate_report(evidence_dir)
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "suite_complete": self.suite_complete,
            "gate_6": "PASS" if self.gate_pass() else "FAIL",
            "items": {
                item.id: {"section": item.section, "spec_line": item.spec_line, **asdict(self.results[item.id])}
                for item in CHECKLIST_ITEMS
            },
        }

    def write_json(self, path: Path, *, evidence_dir: Path | None = None) -> None:
        path.write_text(json.dumps(self.as_dict(evidence_dir), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def write_markdown(self, path: Path, *, evidence_dir: Path | None = None) -> None:
        self.validate_report(evidence_dir)
        lines = [
            f"# Fase 6 E2E — `{self.run_id}`",
            "",
            f"Gate 6: **{'PASS' if self.gate_pass() else 'FAIL'}**",
            "",
            "| ID | Seção | Spec | Status | Observado | Motivo | Evidências |",
            "|---|---|---:|---|---|---|---|",
        ]
        for item in CHECKLIST_ITEMS:
            result = self.results[item.id]
            def cell(value: object) -> str:
                return str(value).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")

            evidence = ", ".join(cell(reference) for reference in result.evidence)
            lines.append(
                f"| {cell(item.id)} | {cell(item.section)} | {item.spec_line} | {cell(result.status)} | "
                f"{cell(result.observed or '—')} | {cell(result.reason or '—')} | {evidence or '—'} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
