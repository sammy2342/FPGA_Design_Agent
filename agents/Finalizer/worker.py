"""
Finalizer / RAG-Archiver agent runtime (AI-enhanced).

Runs ONLY after the RTL pipeline succeeds and:
1) snapshots the final RTL/TB + key logs into a run bundle folder
2) writes a manifest.json for traceability
3) calls an LLM once to produce a structured "design historian" summary (final_summary.json)
4) stores + indexes:
   - exact design memory
   - module/component-level memory
   - testbench pattern memory
   - exact-match spec fingerprint cache

This lets future planner / implementation stages reuse:
- exact passing designs
- partial passing components
- passing TB patterns
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.common.base import AgentWorkerBase
from agents.common.llm_gateway import GenerationConfig, Message, MessageRole, init_llm_gateway
from core.observability.agentops_tracker import get_tracker
from core.observability.emitter import emit_runtime_event
from core.runtime.retry import TaskInputError
from core.schemas.contracts import AgentType, ResultMessage, TaskMessage, TaskStatus

from adapters.rag.rag_service import init_rag_service  # type: ignore


_MODULE_HEADER_WITH_PORTS = re.compile(
    r"\bmodule\s+([A-Za-z_]\w*)\s*(?:#\s*\(.*?\)\s*)?\(\s*(.*?)\s*\)\s*;",
    re.DOTALL,
)
_MODULE_HEADER_NO_PORTS = re.compile(r"\bmodule\s+([A-Za-z_]\w*)\s*;", re.DOTALL)


def _safe_json(text: str) -> Optional[dict]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            return None
    return None


def _read_text(path: Path) -> str:
    try:
        if path and path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return ""


def _sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _parse_attempt(value) -> int | None:
    if value is None:
        return None
    try:
        attempt = int(value)
    except Exception:
        return None
    return attempt if attempt > 0 else None


def _stage_dir(kind: str, attempt: int | None) -> str:
    if attempt is None:
        return kind
    return f"{kind}_attempt{attempt}"


def _extract_signature_and_module_name(rtl_text: str) -> Tuple[Optional[str], Optional[str]]:
    m = _MODULE_HEADER_WITH_PORTS.search(rtl_text)
    if m:
        module_name = m.group(1)
        port_block = " ".join(m.group(2).split())
        signature = f"module {module_name}({port_block});"
        return signature, module_name

    m2 = _MODULE_HEADER_NO_PORTS.search(rtl_text)
    if m2:
        module_name = m2.group(1)
        signature = f"module {module_name};"
        return signature, module_name

    return None, None


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _short_hash(text: str, n: int = 10) -> str:
    return sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def _tail(text: str, n: int) -> str:
    if not text:
        return ""
    return text[-n:]


class FinalizerWorker(AgentWorkerBase):
    handled_types = {AgentType.FINALIZE}
    runtime_name = "agent_finalizer"

    def __init__(self, connection_params, stop_event):
        super().__init__(connection_params, stop_event)
        self.rag = init_rag_service()
        self.gateway = init_llm_gateway()

    # ------------------------------------------------------------------
    # Metadata extraction helpers
    # ------------------------------------------------------------------

    def _classify_design_type(self, module_name: str, rtl_text: str) -> str:
        text = f"{module_name}\n{rtl_text}".lower()

        if "fifo" in text:
            return "fifo"
        if "arbiter" in text:
            return "arbiter"
        if "counter" in text:
            return "counter"
        if re.search(r"\bstate\b|\bcase\b", text):
            return "fsm"
        if "mux" in text:
            return "mux"
        if "decoder" in text:
            return "decoder"
        if "alu" in text:
            return "alu"
        if "shift" in text:
            return "shift_register"
        if re.search(r"\balways_ff\b|\bposedge\b", text):
            return "sequential_logic"
        if re.search(r"\balways_comb\b|\bassign\b", text):
            return "combinational_logic"
        return "generic_module"

    def _extract_behavior_tags(self, module_name: str, rtl_text: str) -> List[str]:
        text = f"{module_name}\n{rtl_text}".lower()
        tags = set()

        if re.search(r"\balways_ff\b|\bposedge\b|\bnegedge\b", text):
            tags.add("sequential")
        if re.search(r"\balways_comb\b|\bassign\b", text):
            tags.add("combinational")
        if re.search(r"\brst\b|\breset\b", text):
            tags.add("reset")
        if re.search(r"\ben\b|\benable\b", text):
            tags.add("enable")
        if re.search(r"\bload\b", text):
            tags.add("load")
        if re.search(r"\bcase\b", text):
            tags.add("case_logic")
        if re.search(r"\bstate\b", text):
            tags.add("stateful")
        if re.search(r"\bclk\b|\bclock\b", text):
            tags.add("clocked")
        if "counter" in text:
            tags.add("counter")
        if "mux" in text:
            tags.add("mux")
        if "decoder" in text:
            tags.add("decoder")
        if re.search(r"\bparameter\b|\blocalparam\b", text):
            tags.add("parameterized")

        return sorted(tags)

    def _extract_interface_features(self, rtl_text: str, ctx_interface: Optional[dict]) -> List[dict]:
        if ctx_interface and isinstance(ctx_interface.get("signals"), list):
            out = []
            for s in ctx_interface["signals"]:
                out.append(
                    {
                        "name": s.get("name", "unknown"),
                        "direction": s.get("direction", "unknown"),
                        "width": str(s.get("width", 1)),
                    }
                )
            return out

        header_match = re.search(
            r"\bmodule\s+[A-Za-z_]\w*\b\s*(?:#\s*\(.*?\)\s*)?\((.*?)\)\s*;",
            rtl_text,
            flags=re.DOTALL,
        )
        if not header_match:
            return []

        port_block = header_match.group(1)
        raw_parts = [p.strip() for p in port_block.split(",") if p.strip()]
        features: List[dict] = []

        for raw in raw_parts:
            direction = "unknown"
            if re.search(r"\binput\b", raw):
                direction = "input"
            elif re.search(r"\boutput\b", raw):
                direction = "output"
            elif re.search(r"\binout\b", raw):
                direction = "inout"

            width_match = re.search(r"\[([^\]]+)\]", raw)
            width = width_match.group(1).strip() if width_match else "1"

            tokens = raw.replace("[", " [").replace("]", "] ").split()
            name = tokens[-1].strip(",;") if tokens else "unknown"

            features.append(
                {
                    "name": name,
                    "direction": direction,
                    "width": width,
                }
            )

        return features

    def _extract_dependencies(self, rtl_text: str, module_name: str) -> List[str]:
        keywords = {
            "module",
            "if",
            "else",
            "for",
            "while",
            "case",
            "always_comb",
            "always_ff",
            "always",
            "assign",
            "begin",
            "end",
            "endmodule",
            "logic",
            "wire",
            "reg",
            "parameter",
            "localparam",
            "generate",
            "endgenerate",
        }

        deps = set()
        inst_pattern = r"\b([A-Za-z_]\w*)\b\s*(?:#\s*\(.*?\)\s*)?\b([A-Za-z_]\w*)\b\s*\("

        for match in re.finditer(inst_pattern, rtl_text, flags=re.DOTALL):
            cand = match.group(1)
            if cand == module_name or cand in keywords:
                continue
            deps.add(cand)

        return sorted(deps)

    def _extract_tb_pattern_tags(self, tb_text: str) -> List[str]:
        text = tb_text.lower()
        tags = set()

        if "clock" in text or re.search(r"\bclk\b", text):
            tags.add("clock_gen")
        if "reset" in text or re.search(r"\brst\b", text):
            tags.add("reset_sequence")
        if re.search(r"\$display|\$monitor", text):
            tags.add("output_observation")
        if re.search(r"\$finish", text):
            tags.add("termination_control")
        if re.search(r"\bassert\b", text):
            tags.add("assertion_based")
        if re.search(r"\brepeat\b|\bfor\b", text):
            tags.add("iterative_stimulus")
        if re.search(r"\btask\b", text):
            tags.add("task_based_tb")

        return sorted(tags)

    def _compute_verification_confidence(
        self,
        lint_log: str,
        tb_lint_log: str,
        sim_log: str,
        reflection: str,
        verification: Optional[dict],
    ) -> float:
        score = 0.0

        if lint_log.strip():
            score += 0.2
            if "error" not in lint_log.lower():
                score += 0.1

        if tb_lint_log.strip():
            score += 0.15
            if "error" not in tb_lint_log.lower():
                score += 0.1

        if sim_log.strip():
            score += 0.2
            low = sim_log.lower()
            if any(x in low for x in ["pass", "success", "completed"]):
                score += 0.15

        if verification and isinstance(verification, dict):
            score += 0.1
            goals = verification.get("goals")
            if isinstance(goals, list) and goals:
                score += min(0.1, len(goals) * 0.02)

        if reflection.strip():
            score += 0.05

        return round(min(score, 1.0), 3)

    def _extract_verification_goals(self, verification: Optional[dict], ai_summary: Optional[dict]) -> List[str]:
        goals: List[str] = []

        if verification and isinstance(verification, dict):
            raw_goals = verification.get("goals")
            if isinstance(raw_goals, list):
                goals.extend([str(x).strip() for x in raw_goals if str(x).strip()])

        if ai_summary and isinstance(ai_summary.get("verification_strategy"), list):
            goals.extend(
                [str(x).strip() for x in ai_summary["verification_strategy"] if str(x).strip()]
            )

        seen = set()
        out = []
        for g in goals:
            if g.lower() not in seen:
                seen.add(g.lower())
                out.append(g)
        return out

    def _extract_original_spec(self, ctx: Dict[str, Any], node_id: str, run_id: str) -> str:
        """
        Try hard to recover the real original design request/spec from pipeline context.
        Falls back to archive text only if nothing useful exists.
        """
        direct_keys = [
            "user_input",
            "prompt",
            "spec",
            "design_request",
            "request",
            "task_description",
            "description",
            "problem",
            "original_prompt",
            "original_spec",
            "planner_prompt",
            "planner_input",
            "goal",
        ]

        for key in direct_keys:
            value = ctx.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        nested_dict_keys = [
            "planner",
            "planning",
            "plan",
            "input",
            "inputs",
            "request_context",
            "design_context",
            "metadata",
        ]

        for outer_key in nested_dict_keys:
            outer_val = ctx.get(outer_key)
            if not isinstance(outer_val, dict):
                continue
            for key in direct_keys:
                value = outer_val.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        # final fallback
        return f"ARCHIVE FINAL PASSING DESIGN node={node_id} run_id={run_id}"

    # ------------------------------------------------------------------
    # Main task
    # ------------------------------------------------------------------

    def handle_task(self, task: TaskMessage) -> ResultMessage:
        ctx = task.context or {}

        if "node_id" not in ctx:
            raise TaskInputError("Missing node_id in task context.")
        if "rtl_path" not in ctx:
            raise TaskInputError("Missing rtl_path in task context.")

        node_id = str(ctx["node_id"])
        attempt = _parse_attempt(ctx.get("attempt"))

        rtl_path = Path(ctx["rtl_path"])
        tb_path = Path(ctx.get("tb_path", "")) if ctx.get("tb_path") else rtl_path.with_name(f"{node_id}_tb.sv")

        rtl_text = _read_text(rtl_path)
        tb_text = _read_text(tb_path)

        if not rtl_text.strip():
            return ResultMessage(
                task_id=task.task_id,
                correlation_id=task.correlation_id,
                status=TaskStatus.FAILURE,
                log_output=f"Finalizer missing RTL contents at {rtl_path}",
            )

        if not tb_text.strip():
            return ResultMessage(
                task_id=task.task_id,
                correlation_id=task.correlation_id,
                status=TaskStatus.FAILURE,
                log_output=f"Finalizer missing TB contents at {tb_path}",
            )

        signature, module_name = _extract_signature_and_module_name(rtl_text)
        if not module_name:
            module_name = node_id
        if not signature:
            signature = f"module {module_name}(/* unknown */);"

        task_memory_root = Path("artifacts/task_memory") / node_id
        sim_log = _read_text(task_memory_root / _stage_dir("sim", attempt) / "log.txt")
        lint_log = _read_text(task_memory_root / _stage_dir("lint", attempt) / "log.txt")
        tb_lint_log = _read_text(task_memory_root / _stage_dir("tb_lint", attempt) / "log.txt")
        reflection = _read_text(task_memory_root / _stage_dir("reflect", attempt) / "reflection_insights.json")
        distilled = _read_text(task_memory_root / _stage_dir("distill", attempt) / "distilled_dataset.json")

        rtl_sha = _sha256_text(rtl_text)
        tb_sha = _sha256_text(tb_text)

        ts = _utc_now_compact()
        run_id = f"{node_id}__{ts}__attempt{attempt if attempt is not None else 'x'}__{_short_hash(rtl_sha + tb_sha)}"
        bundle_root = Path("artifacts/rag_runs") / node_id / run_id
        logs_dir = bundle_root / "logs"
        insights_dir = bundle_root / "insights"
        bundle_root.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        insights_dir.mkdir(parents=True, exist_ok=True)

        (bundle_root / "rtl.sv").write_text(rtl_text, encoding="utf-8")
        (bundle_root / "tb.sv").write_text(tb_text, encoding="utf-8")

        interface = None
        if isinstance(ctx.get("interface"), dict) and isinstance(ctx["interface"].get("signals"), list):
            interface = {"signals": ctx["interface"]["signals"]}
            (bundle_root / "interface.json").write_text(json.dumps(interface, indent=2), encoding="utf-8")

        verification = ctx.get("verification") if isinstance(ctx.get("verification"), dict) else None
        if verification is not None:
            (bundle_root / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")

        if lint_log.strip():
            (logs_dir / "lint.log").write_text(lint_log, encoding="utf-8")
        if tb_lint_log.strip():
            (logs_dir / "tb_lint.log").write_text(tb_lint_log, encoding="utf-8")
        if sim_log.strip():
            (logs_dir / "sim.log").write_text(sim_log, encoding="utf-8")
        if distilled.strip():
            (insights_dir / "distilled_dataset.json").write_text(distilled, encoding="utf-8")
        if reflection.strip():
            (insights_dir / "reflection_insights.json").write_text(reflection, encoding="utf-8")

        ai_summary: Optional[dict] = None
        ai_log = "LLM summary skipped (gateway disabled/unavailable)."

        if self.gateway and os.getenv("USE_LLM", "1") == "1":
            ai_summary = self._llm_summarize(
                node_id=node_id,
                module_name=module_name,
                signature=signature,
                rtl_text=rtl_text,
                tb_text=tb_text,
                sim_log=sim_log,
                reflection=reflection,
                verification=verification,
                interface_signals=(interface or {}).get("signals") if interface else None,
            )
            if ai_summary:
                (bundle_root / "final_summary.json").write_text(
                    json.dumps(ai_summary, indent=2),
                    encoding="utf-8",
                )
                ai_log = "LLM summary created (final_summary.json)."
            else:
                ai_log = "LLM summary attempted but invalid/empty JSON; continuing without AI summary."

        summary = str(ctx.get("final_summary") or "").strip()
        if not summary and ai_summary and isinstance(ai_summary.get("design_summary"), str):
            summary = ai_summary["design_summary"].strip()
        if not summary:
            summary = f"{module_name}: Final passing design archived for reuse."

        design_type = self._classify_design_type(module_name, rtl_text)
        behavior_tags = self._extract_behavior_tags(module_name, rtl_text)
        interface_features = self._extract_interface_features(rtl_text, interface)
        dependencies = self._extract_dependencies(rtl_text, module_name)
        verification_confidence = self._compute_verification_confidence(
            lint_log=lint_log,
            tb_lint_log=tb_lint_log,
            sim_log=sim_log,
            reflection=reflection,
            verification=verification,
        )
        verification_goals = self._extract_verification_goals(verification, ai_summary)
        tb_pattern_tags = self._extract_tb_pattern_tags(tb_text)

        original_spec = self._extract_original_spec(ctx, node_id=node_id, run_id=run_id)

        design_metadata = {
            "module_name": module_name,
            "summary": summary,
            "signature": signature,
            "design_type": design_type,
            "behavior_tags": behavior_tags,
            "interface_features": interface_features,
            "dependencies": dependencies,
            "verification_confidence": verification_confidence,
            "verification_goals": verification_goals,
            "original_spec": original_spec,
        }

        tb_metadata = {
            "module_name": module_name,
            "tb_text": tb_text,
            "summary": f"{module_name}: Passing testbench pattern",
            "pattern_tags": tb_pattern_tags,
            "verification_confidence": verification_confidence,
        }

        rag_memory_file = os.getenv("RAG_MEMORY_FILE", "verilog_rag_memory.json")
        rag_log = "RAG disabled or unavailable; skipping indexing."
        inserted = None

        if self.rag is not None and os.getenv("USE_RAG", "0") == "1":
            assistant_output_for_rag = (
                f"{summary}\n"
                f"{json.dumps(ai_summary, indent=2) if ai_summary else ''}\n\n"
                f"{rtl_text}\n"
            )
            user_input_for_rag = original_spec

            try:
                inserted = self.rag.update_memory(
                    user_input_for_rag,
                    assistant_output_for_rag,
                    design_metadata=design_metadata,
                    tb_metadata=tb_metadata,
                )
            except Exception as e:
                rag_log = f"RAG indexing error: {type(e).__name__}: {e}"
            else:
                rag_log = (
                    f"RAG stored designs={inserted.get('designs', [])}, "
                    f"components={inserted.get('components', [])}, "
                    f"tb_patterns={inserted.get('tb_patterns', [])}"
                )
        elif os.getenv("USE_RAG", "0") != "1":
            rag_log = "RAG disabled (USE_RAG!=1); skipping indexing."

        manifest: Dict[str, Any] = {
            "schema_version": "2.0",
            "run_id": run_id,
            "node_id": node_id,
            "timestamp_utc": ts,
            "attempt": attempt,
            "dut_name": module_name,
            "signature": signature,
            "summary": summary,
            "original_spec": original_spec,
            "hashes": {"rtl_sha256": rtl_sha, "tb_sha256": tb_sha},
            "bundle_root": str(bundle_root),
            "ai_summary_path": str(bundle_root / "final_summary.json")
            if (bundle_root / "final_summary.json").exists()
            else None,
            "design_metadata": {
                "design_type": design_type,
                "behavior_tags": behavior_tags,
                "interface_features": interface_features,
                "dependencies": dependencies,
                "verification_goals": verification_goals,
                "verification_confidence": verification_confidence,
                "original_spec": original_spec,
            },
            "tb_metadata": {
                "pattern_tags": tb_pattern_tags,
            },
            "artifacts": {
                "rtl_path": str(bundle_root / "rtl.sv"),
                "tb_path": str(bundle_root / "tb.sv"),
                "logs": {
                    "lint": str(logs_dir / "lint.log") if (logs_dir / "lint.log").exists() else None,
                    "tb_lint": str(logs_dir / "tb_lint.log") if (logs_dir / "tb_lint.log").exists() else None,
                    "sim": str(logs_dir / "sim.log") if (logs_dir / "sim.log").exists() else None,
                },
                "insights": {
                    "distilled": str(insights_dir / "distilled_dataset.json")
                    if (insights_dir / "distilled_dataset.json").exists()
                    else None,
                    "reflection": str(insights_dir / "reflection_insights.json")
                    if (insights_dir / "reflection_insights.json").exists()
                    else None,
                },
                "interface": str(bundle_root / "interface.json") if (bundle_root / "interface.json").exists() else None,
                "verification": str(bundle_root / "verification.json")
                if (bundle_root / "verification.json").exists()
                else None,
            },
            "outcome": {"status": "SUCCESS"},
            "notes": {
                "ai": ai_log,
                "rag": rag_log,
                "rag_memory_file": rag_memory_file,
            },
        }
        (bundle_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        try:
            extra_record = {
                "module_name": module_name,
                "signature": signature,
                "summary": summary,
                "original_spec": original_spec,
                "run_id": run_id,
                "node_id": node_id,
                "attempt": attempt,
                "timestamp_utc": ts,
                "rtl_sha256": rtl_sha,
                "tb_sha256": tb_sha,
                "design_type": design_type,
                "behavior_tags": behavior_tags,
                "interface_features": interface_features,
                "dependencies": dependencies,
                "verification_goals": verification_goals,
                "verification_confidence": verification_confidence,
                "tb_pattern_tags": tb_pattern_tags,
                "ai_log": ai_log,
                "rag_log": rag_log,
                "rag_memory_file_env": rag_memory_file,
                "inserted": inserted,
                "ai_summary": ai_summary,
                "bundle_root": str(bundle_root),
            }
            (bundle_root / "rag_record.json").write_text(json.dumps(extra_record, indent=2), encoding="utf-8")
        except Exception:
            pass

        emit_runtime_event(
            runtime=self.runtime_name,
            event_type="task_completed",
            payload={"task_id": str(task.task_id), "run_id": run_id, "bundle_root": str(bundle_root)},
        )

        return ResultMessage(
            task_id=task.task_id,
            correlation_id=task.correlation_id,
            status=TaskStatus.SUCCESS,
            artifacts_path=str(bundle_root),
            log_output=f"Finalizer archived run_id={run_id}. {ai_log} {rag_log}",
            reflections=json.dumps(
                {
                    "run_id": run_id,
                    "bundle_root": str(bundle_root),
                    "dut_name": module_name,
                    "signature": signature,
                    "rtl_sha256": rtl_sha,
                    "tb_sha256": tb_sha,
                    "design_type": design_type,
                    "behavior_tags": behavior_tags,
                    "verification_confidence": verification_confidence,
                    "original_spec": original_spec,
                    "ai": ai_log,
                    "rag": rag_log,
                    "rag_memory_file": rag_memory_file,
                },
                indent=2,
            ),
        )

    # ------------------------------------------------------------------
    # LLM summary
    # ------------------------------------------------------------------

    def _llm_summarize(
        self,
        *,
        node_id: str,
        module_name: str,
        signature: str,
        rtl_text: str,
        tb_text: str,
        sim_log: str,
        reflection: str,
        verification: Optional[dict],
        interface_signals: Optional[list],
    ) -> Optional[dict]:
        if not self.gateway:
            return None

        system = (
            "You are an RTL Finalizer Agent.\n"
            "A design has PASSED simulation. Produce a compact, high-signal summary that helps future agents reuse it.\n"
            "Return ONLY valid JSON (no extra text, no code fences).\n"
            "Schema (exact keys):\n"
            "{\n"
            '  "design_summary": string,\n'
            '  "interface_overview": [{"name": string, "direction": "input"|"output", "width": string}],\n'
            '  "key_behaviors": [string],\n'
            '  "verification_strategy": [string],\n'
            '  "reusable_patterns": [string],\n'
            '  "assumptions_and_limits": [string]\n'
            "}\n"
            "Rules:\n"
            "- Do NOT invent ports; infer from RTL only.\n"
            "- Keep each list item <= 1 sentence.\n"
            "- If logs/insights are missing, mention that in assumptions_and_limits.\n"
            "- Focus on what matters for reuse.\n"
        )

        rtl_clip = rtl_text if len(rtl_text) <= 12000 else rtl_text[:12000] + "\n// [truncated]\n"
        tb_clip = tb_text if len(tb_text) <= 12000 else tb_text[:12000] + "\n// [truncated]\n"
        sim_tail = _tail(sim_log, 4000)
        refl_tail = _tail(reflection, 4000)

        user = (
            f"node_id: {node_id}\n"
            f"module_name: {module_name}\n"
            f"signature: {signature}\n\n"
            f"known_interface_signals (may be partial): {json.dumps(interface_signals or [], indent=2)}\n\n"
            f"verification_context (may be partial): {json.dumps(verification or {}, indent=2)}\n\n"
            "FINAL RTL:\n"
            f"{rtl_clip}\n\n"
            "FINAL TESTBENCH:\n"
            f"{tb_clip}\n\n"
            "SIM LOG (tail):\n"
            f"{sim_tail}\n\n"
            "REFLECTION INSIGHTS (tail):\n"
            f"{refl_tail}\n"
        )

        msgs = [
            Message(role=MessageRole.SYSTEM, content=system),
            Message(role=MessageRole.USER, content=user),
        ]

        max_tokens = int(os.getenv("LLM_MAX_TOKENS_FINALIZER", "2500"))
        temperature = float(os.getenv("LLM_TEMPERATURE_FINALIZER", "0.2"))
        cfg = GenerationConfig(temperature=temperature, max_tokens=max_tokens)

        try:
            resp = asyncio.run(self.gateway.generate(messages=msgs, config=cfg))  # type: ignore[arg-type]
        except Exception:
            return None

        try:
            tracker = get_tracker()
            tracker.log_llm_call(
                agent=self.runtime_name,
                node_id=node_id,
                model=getattr(resp, "model_name", "unknown"),
                provider=getattr(resp, "provider", "unknown"),
                prompt_tokens=getattr(resp, "input_tokens", 0),
                completion_tokens=getattr(resp, "output_tokens", 0),
                total_tokens=getattr(resp, "total_tokens", 0),
                estimated_cost_usd=getattr(resp, "estimated_cost_usd", None),
                metadata={"stage": "finalizer_summary"},
            )
        except Exception:
            pass

        parsed = _safe_json(getattr(resp, "content", "") or "")
        if not parsed:
            return None

        if not isinstance(parsed.get("design_summary", ""), str):
            return None
        if not isinstance(parsed.get("key_behaviors", []), list):
            return None
        if not isinstance(parsed.get("verification_strategy", []), list):
            return None

        return parsed