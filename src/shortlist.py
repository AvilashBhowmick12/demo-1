"""Resume shortlisting: rank candidates (one row per resume) for a job description.

Score = keyword score  (70%): how many of the wanted skills appear (coverage) and how
                              much each is mentioned (depth), from the full resume text
        + semantic score (30%): how close the resume's best chunks are to the query,
                              ranked against all other resumes (0..1 percentile).

Keywords come in two kinds:
  must_have     - skills the candidate is expected to have (weight 2)
  nice_to_have  - bonus skills (weight 1)
A keyword may list alternatives with "|", e.g. "kubernetes|k8s". Some common
synonyms are built in (see SYNONYMS).
"""
import bisect
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.experience import estimate_experience
from src.names import extract_name

# Spellings that should count as the same skill. Every name in a group matches any other.
SYNONYMS = [
    ["kubernetes", "k8s"],
    ["javascript", "js"],
    ["postgresql", "postgres"],
    ["aws", "amazon web services"],
    ["gcp", "google cloud platform", "google cloud"],
    ["ci/cd", "cicd", "ci-cd"],
    ["scikit-learn", "sklearn", "scikit learn"],
    ["node.js", "nodejs", "node js"],
    ["power bi", "powerbi"],
    ["machine learning", "ml", "apprentissage automatique"],
    ["ai", "artificial intelligence", "intelligence artificielle", "ia"],
    ["llm", "large language model"],
    ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    ["nlp", "natural language processing"],
    ["genai", "generative ai"],
]
_SYN = {name: group for group in SYNONYMS for name in group}

W_MUST, W_NICE = 2.0, 1.0          # weight of a must-have / nice-to-have keyword
W_KEYWORD, W_SEMANTIC = 0.7, 0.3   # balance of the final score (when keywords are given)
KW_COVERAGE, KW_DEPTH = 0.7, 0.3   # inside the keyword score: skills present vs. how often mentioned
DEPTH_SATURATION = 20              # this many mentions of one skill counts as "full depth"
SEM_TOP_CHUNKS = 3                 # semantic score uses the mean of a resume's best N chunks


def expand_term(term: str) -> List[str]:
    """'kubernetes|k8s' -> ['kubernetes', 'k8s'] (plus built-in synonyms)."""
    names: List[str] = []
    for alt in term.split("|"):
        alt = alt.strip().lower()
        if not alt:
            continue
        for n in _SYN.get(alt, [alt]):
            if n not in names:
                names.append(n)
    return names


def _compile(names: Iterable[str]) -> List["re.Pattern"]:
    patterns = []
    for n in names:
        body = re.escape(n).replace(r"\ ", r"\s+")
        if len(n) >= 3 and n[-1].isalpha():
            body += "s?"                      # plural: pipeline(s), LLM(s)
        # not glued to other letters/digits; lets "c++", ".net", "node.js" work
        patterns.append(re.compile(r"(?<![A-Za-z0-9+#])" + body + r"(?![A-Za-z0-9+#])", re.I))
    return patterns


def count_mentions(text: str, patterns: List["re.Pattern"]) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def _depth(count: int) -> float:
    """0..1 for how much a skill is mentioned; grows quickly at first, then flattens."""
    return min(1.0, math.log1p(count) / math.log1p(DEPTH_SATURATION))


def _candidate_id(meta: dict) -> str:
    return meta.get("candidate_id") or Path(meta.get("source", "unknown")).stem


def _read_via(docs) -> str:
    """How a file's text was obtained, for the /add message (OCR is only used for scanned pages)."""
    kind = docs[0].metadata.get("file_type", "")
    if any(d.metadata.get("ocr") for d in docs):
        return "OCR (scanned pages)"
    if kind == "pdf":
        return "PDF text layer (no OCR needed)"
    if kind == "docx":
        return "Word file, read directly (no OCR needed)"
    return "text file, read directly (no OCR needed)"


class ResumeShortlister:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        only_dirs: Iterable[str] = ("resume",),
        store=None,
    ):
        self.persist_dir = Path(persist_dir)
        self.embedding_model = embedding_model
        self.only_dirs = {d.lower() for d in only_dirs}
        self._store = store
        self.candidates: Dict[str, dict] = {}   # candidate_id -> {"source_file", "text"}

    # ---------- storage ----------
    @property
    def store(self):
        if self._store is None:
            from src.vectorstore import FaissVectorStore
            self._store = FaissVectorStore(str(self.persist_dir), self.embedding_model)
        return self._store

    @property
    def _candidates_file(self) -> Path:
        return self.persist_dir / "candidates.json"

    @property
    def _info_file(self) -> Path:
        return self.persist_dir / "index_info.json"

    def _saved_model(self) -> str:
        """Which embedding model built the saved index (indexes from before this check used the default)."""
        try:
            return json.loads(self._info_file.read_text(encoding="utf-8"))["embedding_model"]
        except Exception:
            return "all-MiniLM-L6-v2"

    def _is_resume(self, doc) -> bool:
        parts = [p.lower() for p in Path(doc.metadata.get("source", "")).parts[:-1]]
        return any(p in self.only_dirs for p in parts)

    def build(self, data_dir: str = "data") -> None:
        """Read all resumes, embed them, and save the index plus the per-candidate full text."""
        from src.data_loader import load_all_documents
        docs = [d for d in load_all_documents(data_dir) if self._is_resume(d)]
        if not docs:
            raise ValueError(f"No resumes found under '{data_dir}' (looked in folders named {sorted(self.only_dirs)}).")

        pages: Dict[str, list] = {}
        files: Dict[str, str] = {}
        for d in docs:
            cid = _candidate_id(d.metadata)
            pages.setdefault(cid, []).append((d.metadata.get("page") or 0, d.page_content))
            files[cid] = d.metadata.get("source_file") or Path(d.metadata.get("source", "")).name
        self.candidates = {
            cid: {"source_file": files[cid], "text": "\n".join(t for _, t in sorted(p))}
            for cid, p in pages.items()
        }
        self._add_experience()
        self._add_names()
        self.store.build_from_documents(docs)   # also saves faiss.index + metadata.pkl
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._candidates_file.write_text(json.dumps(self.candidates, ensure_ascii=False), encoding="utf-8")
        self._info_file.write_text(json.dumps({"embedding_model": self.embedding_model}), encoding="utf-8")
        print(f"[INFO] Shortlister built: {len(self.candidates)} candidates")

    def load(self) -> None:
        self.store.load()
        self.candidates = json.loads(self._candidates_file.read_text(encoding="utf-8"))
        self._add_experience()
        self._add_names()
        print(f"[INFO] Shortlister loaded: {len(self.candidates)} candidates")

    def _add_experience(self) -> None:
        """Estimate years of experience for every candidate (cheap, so recomputed on every load)."""
        for info in self.candidates.values():
            e = estimate_experience(info["text"])
            info["years"], info["years_source"] = e["years"], e["source"]
            info["years_stated"], info["years_dates"] = e["stated"], e["dates"]

    def _add_names(self) -> None:
        """Extract each candidate's name; a name shared by 4+ resumes is flagged as a placeholder."""
        for info in self.candidates.values():
            n = extract_name(info["text"], info.get("source_file", ""))
            info["name"], info["name_source"], info["name_placeholder"] = n["name"], n["source"], n["placeholder"]
        counts: Dict[str, int] = {}
        for info in self.candidates.values():
            if info["name"]:
                counts[info["name"].lower()] = counts.get(info["name"].lower(), 0) + 1
        for info in self.candidates.values():
            if info["name"] and counts[info["name"].lower()] >= 4:
                info["name_placeholder"] = True

    def ensure(self, data_dir: str = "data") -> None:
        """Load the saved index if it exists, otherwise build it (slow the first time)."""
        needed = ["faiss.index", "metadata.pkl", "candidates.json"]
        if not all((self.persist_dir / f).exists() for f in needed):
            self.build(data_dir)
        elif self._saved_model() != self.embedding_model:
            print(f"[INFO] Saved index was built with '{self._saved_model()}', not '{self.embedding_model}': rebuilding.")
            self._store = None                      # drop the store holding the old model/index
            self.build(data_dir)
        else:
            self.load()

    # ---------- ranking ----------
    def _semantic_scores(self, query: str) -> Dict[str, dict]:
        """Per candidate: raw cosine of its best chunks, a pool-relative percentile, and a snippet."""
        hits = self.store.query(query, top_k=self.store.index.ntotal)
        sims: Dict[str, list] = {}
        snippet: Dict[str, tuple] = {}
        for h in hits:
            meta = h.get("metadata") or {}
            cid = meta.get("candidate_id")
            if cid is None:
                continue
            cos = 1.0 - float(h["distance"]) / 2.0          # vectors are L2-normalised
            sims.setdefault(cid, []).append(cos)
            if cid not in snippet or cos > snippet[cid][0]:
                snippet[cid] = (cos, meta.get("text", ""))

        raw = {cid: sum(sorted(v, reverse=True)[:SEM_TOP_CHUNKS]) / min(len(v), SEM_TOP_CHUNKS)
               for cid, v in sims.items()}
        ordered = sorted(raw.values())
        n = len(ordered)
        out: Dict[str, dict] = {}
        for cid, r in raw.items():
            below = bisect.bisect_left(ordered, r)           # how many candidates score lower
            out[cid] = {"raw": r, "sim": below / (n - 1) if n > 1 else 1.0, "snippet": snippet[cid][1]}
        return out

    def shortlist(
        self,
        query: str = "",
        must_have: Iterable[str] = (),
        nice_to_have: Iterable[str] = (),
        top_n: int = 10,
        strict: bool = False,
        min_years: Optional[float] = None,
        max_years: Optional[float] = None,
    ) -> List[dict]:
        """Return the top_n candidates, best first.

        Candidates that mention none of the given keywords are left out.
        strict=True also drops every candidate that lacks any must-have keyword.
        min_years / max_years keep only candidates whose estimated experience fits (resumes with
        no estimate are kept and marked years=None, unless strict=True). A date-based estimate gets
        1 year of tolerance; a stated figure gets none.
        """
        must = list(must_have)
        nice = list(nice_to_have)
        if not query:
            query = " ".join(must + nice)
        if not query and not self.candidates:
            return []

        must_pat = {t: _compile(expand_term(t)) for t in must}
        nice_pat = {t: _compile(expand_term(t)) for t in nice}
        has_keywords = bool(must or nice)
        total_weight = W_MUST * len(must) + W_NICE * len(nice)
        semantic = self._semantic_scores(query)

        results = []
        for cid, info in self.candidates.items():
            text = info["text"]
            must_counts = {t: count_mentions(text, p) for t, p in must_pat.items()}
            nice_counts = {t: count_mentions(text, p) for t, p in nice_pat.items()}
            matched_must = [t for t, c in must_counts.items() if c]
            missing_must = [t for t, c in must_counts.items() if not c]
            matched_nice = [t for t, c in nice_counts.items() if c]
            if strict and missing_must:
                continue
            if has_keywords and not (matched_must or matched_nice):
                continue                       # mentions none of the wanted skills
            years, years_src = info.get("years"), info.get("years_source")
            if min_years is not None or max_years is not None:
                if years is None:
                    if strict:
                        continue
                else:
                    tol = 0.0 if years_src == "stated" else 1.0
                    if min_years is not None and years + tol < min_years:
                        continue
                    if max_years is not None and years - tol > max_years:
                        continue

            if total_weight:
                coverage = (W_MUST * len(matched_must) + W_NICE * len(matched_nice)) / total_weight
                depth = (sum(W_MUST * _depth(c) for c in must_counts.values())
                         + sum(W_NICE * _depth(c) for c in nice_counts.values())) / total_weight
                kw = KW_COVERAGE * coverage + KW_DEPTH * depth
            else:
                kw = 0.0
            sem = semantic.get(cid, {"raw": 0.0, "sim": 0.0, "snippet": ""})
            score = (W_KEYWORD * kw + W_SEMANTIC * sem["sim"]) if has_keywords else sem["sim"]
            results.append({
                "candidate_id": cid,
                "source_file": info["source_file"],
                "name": info.get("name"),
                "name_source": info.get("name_source"),
                "name_placeholder": info.get("name_placeholder", False),
                "score": round(100 * score, 1),
                "keyword_score": round(100 * kw, 1),
                "semantic_score": round(100 * sem["sim"], 1),
                "semantic_raw": round(sem["raw"], 3),
                "matched_must": matched_must,
                "missing_must": missing_must,
                "matched_nice": matched_nice,
                "mentions": {**must_counts, **nice_counts},
                "years": years,
                "years_source": years_src,
                "snippet": sem["snippet"][:300],
            })

        results.sort(key=lambda r: (-r["score"], -sum(r["mentions"].values())))
        return results[:top_n]

### when new resume is added to the model 22-11-26
    def add_resume(self, file_path: str, data_dir: str = "data") -> dict:
        """Add ONE new resume to the loaded index without rebuilding (seconds, not hours)."""
        from src.data_loader import load_single_document
        path = Path(file_path.strip().strip('"'))
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if not self.candidates:
            self.load()
        cid = path.stem
        if cid in self.candidates:
            return {"status": "already indexed", "candidate_id": cid}
        docs = load_single_document(path, cache_dir=Path(data_dir) / "ocr_cache")
        if not docs:
            return {"status": "no text found (blank or unreadable file)", "candidate_id": cid}
        read_via = _read_via(docs)
        chunks = self.store.add_documents(docs)      # embeds, then saves faiss.index + metadata.pkl
        pages = sorted((d.metadata.get("page") or 0, d.page_content) for d in docs)
        self.candidates[cid] = {"source_file": path.name, "text": "\n".join(t for _, t in pages)}
        self._add_experience()
        self._add_names()
        self._candidates_file.write_text(json.dumps(self.candidates, ensure_ascii=False), encoding="utf-8")
        c = self.candidates[cid]
        return {"status": "added", "candidate_id": cid, "name": c["name"],
                "chunks": chunks, "years": c["years"], "read_via": read_via}