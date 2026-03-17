from __future__ import annotations

import hashlib
import json
import os
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core import Document, PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except Exception:
    OpenAIEmbedding = None  # type: ignore[misc, assignment]

try:
    from llama_index.embeddings.ollama import OllamaEmbedding
except Exception:
    OllamaEmbedding = None  # type: ignore[misc, assignment]


SHORT_TERM_WINDOW = 8

DEFAULT_MEMORY_FILE = "verilog_rag_memory.json"
DEFAULT_KNOWLEDGE_BASE_FILE = "verilog_knowledge_base.txt"

RANK_TOPIC_BOOST = {
    "exact_design": 2.4,
    "stored_design": 1.8,
    "stored_component": 1.5,
    "stored_tb_pattern": 1.3,
    "verilog_modules": 1.0,
}

CONVERSATIONAL_PROMPT = PromptTemplate(
    """
You are a Verilog/SystemVerilog hardware design assistant.

You have:
- A library of existing modules (from files).
- A library of previously generated/stored designs (long-term memory).
- A library of reusable stored components from successful designs.
- Recent conversation context (short-term memory).

Guidelines:
- Prefer exact passing designs when the request matches.
- Otherwise prefer verified passing components and patterns.
- Reuse only the relevant chunk if the full prior design is not an exact fit.
- Provide working SystemVerilog code (or Verilog if appropriate).
- Follow style from context modules.
- Include clear headers and brief comments.
- For complex tasks, show how modules connect.
- Keep explanations minimal and focused on design clarity.

Recent conversation + user request:
{query_str}

Relevant context from memory / knowledge base:
{context_str}

Now provide the best possible answer:
""".strip()
)


class VerilogRAGService:
    def __init__(
        self,
        knowledge_base_path: Optional[str] = None,
        memory_file_path: Optional[str] = None,
        *,
        embedding_provider: str = "openai",
        openai_api_key: Optional[str] = None,
        openai_embedding_model: str = "text-embedding-3-small",
        ollama_embedding_model: str = "nomic-embed-text",
        rank_topic_boost: Optional[dict] = None,
    ):
        self.rank_topic_boost = rank_topic_boost or dict(RANK_TOPIC_BOOST)

        provider = (embedding_provider or "openai").strip().lower()

        Settings.llm = None  # type: ignore[assignment]

        if provider == "openai":
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "RAG embedding provider is openai but OPENAI_API_KEY is not set."
                )
            if OpenAIEmbedding is None:
                raise RuntimeError(
                    "OpenAIEmbedding not available. Install: llama-index-embeddings-openai"
                )

            Settings.embed_model = OpenAIEmbedding(
                model=openai_embedding_model,
                api_key=api_key,
            )
            print(f"[RAG] Embeddings provider=openai model={openai_embedding_model}")

        elif provider == "ollama":
            if OllamaEmbedding is None:
                raise RuntimeError(
                    "OllamaEmbedding not available. Install: llama-index-embeddings-ollama"
                )
            Settings.embed_model = OllamaEmbedding(model_name=ollama_embedding_model)
            print(f"[RAG] Embeddings provider=ollama model={ollama_embedding_model}")

        else:
            raise RuntimeError(
                f"Unknown RAG_EMBEDDING_PROVIDER='{provider}'. Use 'openai' or 'ollama'."
            )

        self.knowledge_base_path = Path(
            knowledge_base_path or DEFAULT_KNOWLEDGE_BASE_FILE
        )
        self.memory_file_path = Path(memory_file_path or DEFAULT_MEMORY_FILE)

        self.base_docs = self._load_verilog_modules_from_txt(self.knowledge_base_path)

        self.memory_data = self._load_long_term_memory()
        all_docs = self.base_docs + self._memory_to_docs(self.memory_data)

        self.index = VectorStoreIndex.from_documents(all_docs)
        self.short_term_history: deque = deque(maxlen=SHORT_TERM_WINDOW)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _load_long_term_memory(self) -> dict:
        if not self.memory_file_path.exists():
            return {
                "schema_version": "2.0",
                "designs": [],
                "components": [],
                "tb_patterns": [],
                "exact_design_cache": {},
            }

        try:
            with open(self.memory_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError("Memory file must be a dict")

            data.setdefault("schema_version", "2.0")
            data.setdefault("designs", [])
            data.setdefault("components", [])
            data.setdefault("tb_patterns", [])
            data.setdefault("exact_design_cache", {})
            return data

        except Exception:
            return {
                "schema_version": "2.0",
                "designs": [],
                "components": [],
                "tb_patterns": [],
                "exact_design_cache": {},
            }

    def _save_long_term_memory(self) -> None:
        try:
            with open(self.memory_file_path, "w", encoding="utf-8") as f:
                json.dump(self.memory_data, f, indent=2)
        except Exception as e:
            print(f"[RAG Memory] Failed to save long-term memory: {e}")

    # -------------------------------------------------------------------------
    # General helpers
    # -------------------------------------------------------------------------

    def _hash_text(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _normalize_text(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip()).lower()

    def _dedupe_key(self, *parts: str) -> str:
        return self._hash_text("||".join(parts))

    def _safe_json_parse(self, text: str) -> Optional[dict]:
        if not text or not text.strip():
            return None

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    def _extract_json_block(self, text: str) -> Optional[dict]:
        return self._safe_json_parse(text)

    # -------------------------------------------------------------------------
    # RTL parsing / classification
    # -------------------------------------------------------------------------

    def _extract_module_blocks(self, text: str) -> List[Tuple[str, str]]:
        pattern = r"\bmodule\s+([A-Za-z_]\w*)\b.*?\bendmodule\b"
        matches = list(re.finditer(pattern, text, flags=re.DOTALL))
        out: List[Tuple[str, str]] = []
        for m in matches:
            module_name = m.group(1)
            rtl = text[m.start() : m.end()].strip()
            out.append((module_name, rtl))
        return out

    def _extract_module_signature(self, rtl: str, module_name: str) -> str:
        pat = rf"\bmodule\s+{re.escape(module_name)}\b\s*(?:#\s*\(.*?\)\s*)?\(\s*(.*?)\s*\)\s*;"
        m = re.search(pat, rtl, flags=re.DOTALL)
        if m:
            port_block = " ".join(m.group(1).split())
            return f"module {module_name}({port_block});"

        m2 = re.search(rf"\bmodule\s+{re.escape(module_name)}\b\s*;", rtl)
        if m2:
            return f"module {module_name};"

        return f"module {module_name}(/* ports unknown */);"

    def _extract_interface_features(self, rtl: str) -> List[dict]:
        features: List[dict] = []

        header_match = re.search(
            r"\bmodule\s+[A-Za-z_]\w*\b\s*(?:#\s*\(.*?\)\s*)?\((.*?)\)\s*;",
            rtl,
            flags=re.DOTALL,
        )
        if not header_match:
            return features

        port_block = header_match.group(1)
        raw_parts = [p.strip() for p in port_block.split(",") if p.strip()]

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

    def _extract_dependencies(self, rtl: str, module_name: str) -> List[str]:
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
        for match in re.finditer(inst_pattern, rtl, flags=re.DOTALL):
            cand = match.group(1)
            if cand == module_name or cand in keywords:
                continue
            deps.add(cand)

        return sorted(deps)

    def _classify_design_type(self, module_name: str, rtl: str) -> str:
        text = f"{module_name}\n{rtl}".lower()

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

    def _extract_behavior_tags(self, module_name: str, rtl: str) -> List[str]:
        text = f"{module_name}\n{rtl}".lower()
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

    def _component_type_from_metadata(
        self, module_name: str, design_type: str, behavior_tags: List[str]
    ) -> str:
        lower_name = module_name.lower()

        if "tb" in lower_name or "testbench" in lower_name:
            return "testbench"
        if design_type == "counter":
            return "counter_core"
        if design_type == "fsm":
            return "fsm_block"
        if design_type == "mux":
            return "mux_block"
        if "reset" in lower_name or "reset" in behavior_tags:
            return "reset_logic"
        if "combinational" in behavior_tags:
            return "combinational_block"
        if "sequential" in behavior_tags:
            return "sequential_block"
        return "generic_component"

    def _make_summary(self, module_name: str, rtl: str) -> str:
        design_type = self._classify_design_type(module_name, rtl)
        tags = self._extract_behavior_tags(module_name, rtl)

        if design_type == "counter":
            return f"{module_name}: Passing counter design with tags {', '.join(tags)}."
        if design_type == "fsm":
            return f"{module_name}: Passing FSM-like module with tags {', '.join(tags)}."
        if design_type == "mux":
            return f"{module_name}: Passing mux-style module with tags {', '.join(tags)}."
        return f"{module_name}: Passing stored RTL module with tags {', '.join(tags)}."

    # -------------------------------------------------------------------------
    # Fingerprinting / exact match
    # -------------------------------------------------------------------------

    def compute_spec_fingerprint(
        self,
        *,
        module_name: str,
        signature: str,
        design_type: str,
        behavior_tags: List[str],
        interface_features: List[dict],
        verification_goals: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        normalized = {
            "module_name": module_name,
            "signature": self._normalize_text(signature),
            "design_type": design_type,
            "behavior_tags": sorted(set(behavior_tags)),
            "interface_features": sorted(
                [
                    {
                        "name": f.get("name", ""),
                        "direction": f.get("direction", ""),
                        "width": str(f.get("width", "1")),
                    }
                    for f in interface_features
                ],
                key=lambda x: (x["direction"], x["name"], x["width"]),
            ),
            "verification_goals": sorted(
                [self._normalize_text(x) for x in (verification_goals or [])]
            ),
            "dependencies": sorted(dependencies or []),
        }
        blob = json.dumps(normalized, sort_keys=True)
        return self._hash_text(blob)

    def lookup_exact_design(self, spec_fingerprint: str) -> Optional[dict]:
        if not spec_fingerprint:
            return None

        exact_cache = self.memory_data.get("exact_design_cache", {})
        match_key = exact_cache.get(spec_fingerprint)
        if not match_key:
            return None

        for d in self.memory_data.get("designs", []):
            if d.get("design_key") == match_key:
                return d
        return None

    # -------------------------------------------------------------------------
    # Memory conversion to documents
    # -------------------------------------------------------------------------

    def _memory_to_docs(self, memory_data: dict) -> List[Document]:
        docs: List[Document] = []
        max_embed_chars = int(os.getenv("RAG_STORED_RTL_EMBED_MAX_CHARS", "12000"))

        for d in memory_data.get("designs", []):
            rtl = d.get("rtl", "")
            rtl_for_embedding = rtl[:max_embed_chars] if isinstance(rtl, str) else ""

            text = (
                f"// STORED DESIGN\n"
                f"// Module: {d.get('module_name','unknown')}\n"
                f"// Summary: {d.get('summary','')}\n"
                f"// Signature: {d.get('signature','')}\n"
                f"// Design Type: {d.get('design_type','generic_module')}\n"
                f"// Behavior Tags: {', '.join(d.get('behavior_tags', []))}\n"
                f"// Spec Fingerprint: {d.get('spec_fingerprint','')}\n"
                f"// Verification Confidence: {d.get('verification_confidence', 0.0)}\n"
                f"// Dependencies: {', '.join(d.get('dependencies', []))}\n"
                f"// RTL_HASH: {d.get('rtl_hash','')}\n\n"
                f"{rtl_for_embedding}"
            )

            docs.append(
                Document(
                    text=text,
                    metadata={
                        "topic": "exact_design" if d.get("is_exact_cache") else "stored_design",
                        "module_name": d.get("module_name", "unknown"),
                        "rtl_hash": d.get("rtl_hash", ""),
                        "design_type": d.get("design_type", "generic_module"),
                        "behavior_tags": ",".join(d.get("behavior_tags", [])),
                        "verification_confidence": d.get("verification_confidence", 0.0),
                        "spec_fingerprint": d.get("spec_fingerprint", ""),
                        "tags": ",".join(d.get("tags", [])),
                    },
                )
            )

        for c in memory_data.get("components", []):
            rtl = c.get("rtl", "")
            rtl_for_embedding = rtl[:max_embed_chars] if isinstance(rtl, str) else ""

            text = (
                f"// STORED COMPONENT\n"
                f"// Module: {c.get('module_name','unknown')}\n"
                f"// Component Type: {c.get('component_type','generic_component')}\n"
                f"// Summary: {c.get('summary','')}\n"
                f"// Signature: {c.get('signature','')}\n"
                f"// Design Type: {c.get('design_type','generic_module')}\n"
                f"// Behavior Tags: {', '.join(c.get('behavior_tags', []))}\n"
                f"// Dependencies: {', '.join(c.get('dependencies', []))}\n"
                f"// Reusable Standalone: {c.get('reusable_standalone', False)}\n"
                f"// RTL_HASH: {c.get('rtl_hash','')}\n\n"
                f"{rtl_for_embedding}"
            )

            docs.append(
                Document(
                    text=text,
                    metadata={
                        "topic": "stored_component",
                        "module_name": c.get("module_name", "unknown"),
                        "component_type": c.get("component_type", "generic_component"),
                        "rtl_hash": c.get("rtl_hash", ""),
                        "design_type": c.get("design_type", "generic_module"),
                        "behavior_tags": ",".join(c.get("behavior_tags", [])),
                        "verification_confidence": c.get("verification_confidence", 0.0),
                        "tags": ",".join(c.get("tags", [])),
                    },
                )
            )

        for tb in memory_data.get("tb_patterns", []):
            body = (tb.get("tb_text") or "")[:max_embed_chars]
            text = (
                f"// STORED TESTBENCH PATTERN\n"
                f"// Module: {tb.get('module_name','unknown')}\n"
                f"// Summary: {tb.get('summary','')}\n"
                f"// Pattern Tags: {', '.join(tb.get('pattern_tags', []))}\n"
                f"// Verification Goals: {', '.join(tb.get('verification_goals', []))}\n\n"
                f"{body}"
            )

            docs.append(
                Document(
                    text=text,
                    metadata={
                        "topic": "stored_tb_pattern",
                        "module_name": tb.get("module_name", "unknown"),
                        "pattern_tags": ",".join(tb.get("pattern_tags", [])),
                        "verification_confidence": tb.get("verification_confidence", 0.0),
                        "tags": ",".join(tb.get("tags", [])),
                    },
                )
            )

        return docs

    # -------------------------------------------------------------------------
    # Knowledge base load
    # -------------------------------------------------------------------------

    def _load_verilog_modules_from_txt(self, file_path: Path) -> List[Document]:
        if not file_path.exists():
            print(f"[RAG] Warning: Knowledge base file not found: {file_path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            module_pattern = r"// MODULE:\s*(\w+)\s*(module\s+\w+.*?endmodule)"
            matches = re.findall(module_pattern, content, re.DOTALL)

            docs: List[Document] = []
            for module_name, module_code in matches:
                clean_code = module_code.strip()
                docs.append(
                    Document(
                        text=clean_code,
                        metadata={
                            "file_name": str(file_path.name),
                            "topic": "verilog_modules",
                            "tags": module_name,
                            "module_name": module_name,
                        },
                    )
                )

            print(f"[RAG] Loaded {len(docs)} modules from {file_path}")
            return docs

        except Exception as e:
            print(f"[RAG] Error reading knowledge base file: {e}")
            return []

    # -------------------------------------------------------------------------
    # Query / retrieval
    # -------------------------------------------------------------------------

    def _build_augmented_query_for_llm(self, user_input: str) -> str:
        if not self.short_term_history:
            return f"User: {user_input}"

        history_lines = []
        for turn in self.short_term_history:
            history_lines.append(f"User: {turn['user']}")
            history_lines.append(f"Assistant: {turn['assistant']}")

        history_block = "\n".join(history_lines[-2 * SHORT_TERM_WINDOW :])
        return history_block + f"\nUser: {user_input}"

    def _build_context_str_from_nodes(self, nodes: List, max_chars: int = 7000) -> str:
        parts: List[str] = []
        total = 0

        for node in nodes:
            n = node.node if isinstance(node, NodeWithScore) else node
            text = n.get_content()
            md = n.metadata or {}

            header = (
                f"// SOURCE topic={md.get('topic','')} "
                f"module={md.get('module_name', md.get('tags','unknown'))} "
                f"design_type={md.get('design_type','')} "
                f"component_type={md.get('component_type','')} "
                f"verification_confidence={md.get('verification_confidence','')} "
                f"spec_fingerprint={md.get('spec_fingerprint','')} "
                f"rtl_hash={md.get('rtl_hash','')}\n"
            )

            chunk = header + text.strip() + "\n"
            if total + len(chunk) > max_chars:
                break

            parts.append(chunk)
            total += len(chunk)

        return "\n".join(parts).strip()

    def _symbolic_prefilter(
        self,
        query: str,
        design_type: Optional[str] = None,
        behavior_tags: Optional[List[str]] = None,
    ) -> List[dict]:
        out: List[dict] = []
        q = self._normalize_text(query)
        desired_tags = set(behavior_tags or [])

        for d in self.memory_data.get("designs", []):
            if design_type and d.get("design_type") != design_type:
                continue

            d_tags = set(d.get("behavior_tags", []))
            if desired_tags and not desired_tags.intersection(d_tags):
                continue

            text_blob = " ".join(
                [
                    d.get("module_name", ""),
                    d.get("summary", ""),
                    d.get("signature", ""),
                    " ".join(d.get("behavior_tags", [])),
                    d.get("design_type", ""),
                ]
            ).lower()

            if not q or any(tok in text_blob for tok in q.split()):
                out.append(d)

        out.sort(key=lambda x: float(x.get("verification_confidence", 0.0)), reverse=True)
        return out[:5]

    def _rank_nodes(
        self,
        node_scores: List[NodeWithScore],
        top_k: int,
        exact_spec_fingerprint: Optional[str] = None,
    ) -> List[NodeWithScore]:
        def rank_key(nsw: NodeWithScore) -> Tuple[float, float, str]:
            score = getattr(nsw, "score", None)
            sim = float(score) if score is not None else 0.0
            md = nsw.node.metadata or {}
            topic = md.get("topic", "")
            boost = self.rank_topic_boost.get(topic, 1.0)
            verification_confidence = float(md.get("verification_confidence", 0.0) or 0.0)

            exact_bonus = 0.0
            if exact_spec_fingerprint and md.get("spec_fingerprint") == exact_spec_fingerprint:
                exact_bonus = 3.0

            final_score = (sim * boost) + exact_bonus + (0.25 * verification_confidence)
            return (-final_score, -verification_confidence, topic)

        sorted_nodes = sorted(node_scores, key=rank_key)
        return sorted_nodes[:top_k]

    def retrieve_context(
        self,
        query: str,
        top_k: int = 4,
        include_history: bool = True,
        retrieve_multiple: int = 2,
        design_type: Optional[str] = None,
        behavior_tags: Optional[List[str]] = None,
        exact_spec_fingerprint: Optional[str] = None,
    ) -> Tuple[str, List]:
        # Exact-match lookup first
        exact_design = None
        if exact_spec_fingerprint:
            exact_design = self.lookup_exact_design(exact_spec_fingerprint)

        docs: List[Any] = []

        if exact_design:
            exact_doc = Document(
                text=(
                    f"// EXACT MATCH PASSING DESIGN\n"
                    f"// Module: {exact_design.get('module_name','unknown')}\n"
                    f"// Summary: {exact_design.get('summary','')}\n"
                    f"// Signature: {exact_design.get('signature','')}\n\n"
                    f"{exact_design.get('rtl','')}"
                ),
                metadata={
                    "topic": "exact_design",
                    "module_name": exact_design.get("module_name", "unknown"),
                    "rtl_hash": exact_design.get("rtl_hash", ""),
                    "spec_fingerprint": exact_design.get("spec_fingerprint", ""),
                    "design_type": exact_design.get("design_type", ""),
                    "verification_confidence": exact_design.get("verification_confidence", 1.0),
                },
            )
            docs.append(NodeWithScore(node=exact_doc, score=1.0))

        # Symbolic component/design prefilter
        symbolic_matches = self._symbolic_prefilter(
            query=query,
            design_type=design_type,
            behavior_tags=behavior_tags,
        )
        for match in symbolic_matches:
            doc = Document(
                text=(
                    f"// SYMBOLIC MATCH PASSING DESIGN\n"
                    f"// Module: {match.get('module_name','unknown')}\n"
                    f"// Summary: {match.get('summary','')}\n"
                    f"// Signature: {match.get('signature','')}\n\n"
                    f"{match.get('rtl','')}"
                ),
                metadata={
                    "topic": "stored_design",
                    "module_name": match.get("module_name", "unknown"),
                    "rtl_hash": match.get("rtl_hash", ""),
                    "spec_fingerprint": match.get("spec_fingerprint", ""),
                    "design_type": match.get("design_type", ""),
                    "verification_confidence": match.get("verification_confidence", 0.0),
                },
            )
            docs.append(NodeWithScore(node=doc, score=0.92))

        # Embedding retrieval
        candidate_k = max(top_k, top_k * retrieve_multiple)
        retriever = self.index.as_retriever(similarity_top_k=candidate_k)
        node_scores = retriever.retrieve(query)

        if node_scores and not isinstance(node_scores[0], NodeWithScore):
            node_scores = [
                n if isinstance(n, NodeWithScore) else NodeWithScore(node=n, score=1.0)
                for n in node_scores
            ]

        merged = docs + node_scores
        ranked = self._rank_nodes(
            merged,
            top_k=top_k,
            exact_spec_fingerprint=exact_spec_fingerprint,
        )

        context_str = self._build_context_str_from_nodes(ranked)
        return context_str, ranked

    def build_augmented_prompt(
        self,
        user_query: str,
        context_str: Optional[str] = None,
        top_k: int = 4,
        design_type: Optional[str] = None,
        behavior_tags: Optional[List[str]] = None,
        exact_spec_fingerprint: Optional[str] = None,
    ) -> str:
        if context_str is None:
            context_str, _ = self.retrieve_context(
                user_query,
                top_k=top_k,
                design_type=design_type,
                behavior_tags=behavior_tags,
                exact_spec_fingerprint=exact_spec_fingerprint,
            )

        llm_query = self._build_augmented_query_for_llm(user_query)
        prompt = CONVERSATIONAL_PROMPT.format(
            query_str=llm_query,
            context_str=context_str,
        )
        return prompt

    # -------------------------------------------------------------------------
    # Update / insert memory
    # -------------------------------------------------------------------------

    def update_memory(
        self,
        user_input: str,
        assistant_output: str,
        *,
        design_metadata: Optional[dict] = None,
        tb_metadata: Optional[dict] = None,
    ) -> Dict[str, List[str]]:
        """
        Stores:
        - full design record
        - component/module-level records
        - testbench pattern record
        - exact-match fingerprint mapping
        """
        module_blocks = self._extract_module_blocks(assistant_output)
        if not module_blocks:
            return {"designs": [], "components": [], "tb_patterns": []}

        inserted_designs: List[str] = []
        inserted_components: List[str] = []
        inserted_tb_patterns: List[str] = []

        exact_cache = self.memory_data.setdefault("exact_design_cache", {})
        self.memory_data.setdefault("designs", [])
        self.memory_data.setdefault("components", [])
        self.memory_data.setdefault("tb_patterns", [])

        top_design_meta = design_metadata or {}
        verification_goals = top_design_meta.get("verification_goals", []) or []
        verification_confidence = float(top_design_meta.get("verification_confidence", 0.0) or 0.0)

        top_module_name = str(top_design_meta.get("module_name") or module_blocks[0][0])

        for module_name, rtl in module_blocks:
            signature = self._extract_module_signature(rtl, module_name)
            design_type = self._classify_design_type(module_name, rtl)
            behavior_tags = self._extract_behavior_tags(module_name, rtl)
            interface_features = self._extract_interface_features(rtl)
            dependencies = self._extract_dependencies(rtl, module_name)
            rtl_hash = self._hash_text(rtl)
            summary = self._make_summary(module_name, rtl)

            spec_fingerprint = self.compute_spec_fingerprint(
                module_name=module_name,
                signature=signature,
                design_type=design_type,
                behavior_tags=behavior_tags,
                interface_features=interface_features,
                verification_goals=verification_goals,
                dependencies=dependencies,
            )

            design_key = self._dedupe_key(module_name, rtl_hash, spec_fingerprint)

            existing_design_keys = {
                d.get("design_key", "")
                for d in self.memory_data.get("designs", [])
            }
            if design_key not in existing_design_keys:
                design_record = {
                    "design_key": design_key,
                    "module_name": module_name,
                    "summary": summary,
                    "signature": signature,
                    "rtl": rtl,
                    "rtl_hash": rtl_hash,
                    "design_type": design_type,
                    "behavior_tags": behavior_tags,
                    "interface_features": interface_features,
                    "verification_goals": verification_goals,
                    "verification_confidence": verification_confidence,
                    "dependencies": dependencies,
                    "tags": ["generated_design", "passing_design", "rtl_stored"],
                    "spec_fingerprint": spec_fingerprint,
                    "is_exact_cache": module_name == top_module_name,
                }
                self.memory_data["designs"].append(design_record)

                doc_text = (
                    f"// STORED DESIGN\n"
                    f"// Module: {module_name}\n"
                    f"// Summary: {summary}\n"
                    f"// Signature: {signature}\n"
                    f"// Design Type: {design_type}\n"
                    f"// Behavior Tags: {', '.join(behavior_tags)}\n"
                    f"// Dependencies: {', '.join(dependencies)}\n"
                    f"// Spec Fingerprint: {spec_fingerprint}\n"
                    f"// Verification Confidence: {verification_confidence}\n\n"
                    f"{rtl}"
                )
                doc = Document(
                    text=doc_text,
                    metadata={
                        "topic": "exact_design" if module_name == top_module_name else "stored_design",
                        "module_name": module_name,
                        "rtl_hash": rtl_hash,
                        "spec_fingerprint": spec_fingerprint,
                        "design_type": design_type,
                        "behavior_tags": ",".join(behavior_tags),
                        "verification_confidence": verification_confidence,
                        "tags": "generated_design,passing_design,rtl_stored",
                    },
                )
                try:
                    self.index.insert(doc)
                except Exception as e:
                    print(f"[RAG Memory] Failed to index design {module_name}: {e}")

                inserted_designs.append(module_name)

            # exact-match cache only for top-level design
            if module_name == top_module_name:
                exact_cache[spec_fingerprint] = design_key

            component_type = self._component_type_from_metadata(
                module_name=module_name,
                design_type=design_type,
                behavior_tags=behavior_tags,
            )
            component_key = self._dedupe_key("component", module_name, rtl_hash)

            existing_component_keys = {
                c.get("component_key", "")
                for c in self.memory_data.get("components", [])
            }
            if component_key not in existing_component_keys:
                component_record = {
                    "component_key": component_key,
                    "module_name": module_name,
                    "component_type": component_type,
                    "summary": summary,
                    "signature": signature,
                    "rtl": rtl,
                    "rtl_hash": rtl_hash,
                    "design_type": design_type,
                    "behavior_tags": behavior_tags,
                    "interface_features": interface_features,
                    "dependencies": dependencies,
                    "verification_confidence": verification_confidence,
                    "reusable_standalone": len(dependencies) == 0,
                    "tags": ["component_memory", "passing_component", component_type],
                }
                self.memory_data["components"].append(component_record)

                component_doc = Document(
                    text=(
                        f"// STORED COMPONENT\n"
                        f"// Module: {module_name}\n"
                        f"// Component Type: {component_type}\n"
                        f"// Summary: {summary}\n"
                        f"// Signature: {signature}\n"
                        f"// Design Type: {design_type}\n"
                        f"// Behavior Tags: {', '.join(behavior_tags)}\n"
                        f"// Dependencies: {', '.join(dependencies)}\n\n"
                        f"{rtl}"
                    ),
                    metadata={
                        "topic": "stored_component",
                        "module_name": module_name,
                        "component_type": component_type,
                        "rtl_hash": rtl_hash,
                        "design_type": design_type,
                        "behavior_tags": ",".join(behavior_tags),
                        "verification_confidence": verification_confidence,
                        "tags": f"component_memory,passing_component,{component_type}",
                    },
                )
                try:
                    self.index.insert(component_doc)
                except Exception as e:
                    print(f"[RAG Memory] Failed to index component {module_name}: {e}")

                inserted_components.append(module_name)

        # Testbench pattern memory
        if tb_metadata and tb_metadata.get("tb_text"):
            tb_text = str(tb_metadata.get("tb_text", ""))
            pattern_tags = list(tb_metadata.get("pattern_tags", []) or [])
            tb_module_name = str(tb_metadata.get("module_name") or top_module_name)
            tb_summary = str(tb_metadata.get("summary") or f"{tb_module_name}: passing TB pattern")
            tb_key = self._dedupe_key("tb", tb_module_name, self._hash_text(tb_text))

            existing_tb_keys = {
                t.get("tb_key", "")
                for t in self.memory_data.get("tb_patterns", [])
            }
            if tb_key not in existing_tb_keys:
                tb_record = {
                    "tb_key": tb_key,
                    "module_name": tb_module_name,
                    "summary": tb_summary,
                    "tb_text": tb_text,
                    "pattern_tags": pattern_tags,
                    "verification_goals": verification_goals,
                    "verification_confidence": verification_confidence,
                    "tags": ["tb_pattern", "passing_tb_pattern"],
                }
                self.memory_data["tb_patterns"].append(tb_record)

                tb_doc = Document(
                    text=(
                        f"// STORED TESTBENCH PATTERN\n"
                        f"// Module: {tb_module_name}\n"
                        f"// Summary: {tb_summary}\n"
                        f"// Pattern Tags: {', '.join(pattern_tags)}\n"
                        f"// Verification Goals: {', '.join(verification_goals)}\n\n"
                        f"{tb_text}"
                    ),
                    metadata={
                        "topic": "stored_tb_pattern",
                        "module_name": tb_module_name,
                        "verification_confidence": verification_confidence,
                        "tags": "tb_pattern,passing_tb_pattern",
                    },
                )
                try:
                    self.index.insert(tb_doc)
                except Exception as e:
                    print(f"[RAG Memory] Failed to index TB pattern for {tb_module_name}: {e}")

                inserted_tb_patterns.append(tb_module_name)

        self._save_long_term_memory()
        self.short_term_history.append({"user": user_input, "assistant": assistant_output})

        return {
            "designs": inserted_designs,
            "components": inserted_components,
            "tb_patterns": inserted_tb_patterns,
        }

    # -------------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------------

    def get_available_modules(self) -> List[str]:
        return sorted(
            {doc.metadata.get("module_name", "unknown") for doc in self.base_docs}
        )

    def get_stored_designs(self) -> List[str]:
        return sorted(
            {d.get("module_name", "unknown") for d in self.memory_data.get("designs", [])}
        )

    def get_stored_components(self) -> List[str]:
        return sorted(
            {c.get("module_name", "unknown") for c in self.memory_data.get("components", [])}
        )

    def get_exact_fingerprint_count(self) -> int:
        return len(self.memory_data.get("exact_design_cache", {}))


def init_rag_service(
    knowledge_base_path: Optional[str] = None,
    memory_file_path: Optional[str] = None,
) -> Optional[VerilogRAGService]:
    if os.getenv("USE_RAG") != "1":
        return None

    knowledge_base = knowledge_base_path or os.getenv(
        "RAG_KNOWLEDGE_BASE", DEFAULT_KNOWLEDGE_BASE_FILE
    )
    memory_file = memory_file_path or os.getenv("RAG_MEMORY_FILE", DEFAULT_MEMORY_FILE)

    provider = (os.getenv("RAG_EMBEDDING_PROVIDER") or "openai").strip().lower()
    openai_key = os.getenv("OPENAI_API_KEY")

    if provider == "openai" and not openai_key:
        raise RuntimeError(
            "USE_RAG=1 and RAG_EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is missing."
        )

    return VerilogRAGService(
        knowledge_base_path=knowledge_base,
        memory_file_path=memory_file,
        embedding_provider=provider,
        openai_api_key=openai_key or None,
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    )