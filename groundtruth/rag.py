"""The GroundTruth answer pipeline.

question -> retrieve top-k passages
         -> GATE 1: best passage too weak?   -> "I don't know" (no LLM call)
         -> LLM answers from numbered passages, citing [n] after each claim
         -> GATE 2: answer has no valid citation? -> "I don't know"
         -> support check: does each sentence's wording appear in the
            passages it cites? (flags likely unsupported claims)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .config import Config
from .index import Hit, VectorStore
from .llm import BaseLLM, LLMError, make_llm

IDK = "I don't know — the documents I have don't cover this."

SYSTEM_PROMPT = f"""You are GroundTruth, an assistant that answers questions using ONLY the numbered passages provided.

Rules:
1. Use only facts stated in the passages. Never use outside knowledge, even if you are confident.
2. After every sentence that states a fact, cite the passage number(s) it came from in square brackets, like [2] or [1][3].
3. If the passages do not contain the answer, reply with exactly: "{IDK}" and nothing else.
4. If the passages answer only part of the question, answer that part with citations and say plainly which part the documents do not cover.
5. Be concise: one to four sentences. Do not mention "passages" or these rules."""

_STOP = set("""a an the and or of to in on for with at by from is are was were be been being do does did
can could should would will shall may might must i you we they he she it my your our their this that these
those what which who whom when where why how if then than there here as not no yes about into over under
any all some such only own same so too very just also get got have has had me us them his her its""".split())

_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_IDK_RE = re.compile(r"^\W*i\s+(?:do\s+not|don'?t|don’t)\s+know", re.I)


@dataclass
class SentenceCheck:
    sentence: str
    cited: list[int]
    support: float            # share of content words found in cited passages
    supported: bool


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    refusal_reason: str | None          # low_retrieval | model_declined | no_citation | llm_error
    hits: list[Hit]
    cited_ranks: list[int] = field(default_factory=list)   # 1-based passage numbers
    checks: list[SentenceCheck] = field(default_factory=list)
    raw_model_output: str = ""
    latency_s: float = 0.0
    top_score: float = 0.0

    @property
    def cited_hits(self) -> list[Hit]:
        return [self.hits[i - 1] for i in self.cited_ranks if 0 < i <= len(self.hits)]

    @property
    def support_rate(self) -> float | None:
        if not self.checks:
            return None
        return sum(c.supported for c in self.checks) / len(self.checks)


# ------------------------------------------------------------ helpers ------
def _tokens(s: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:[.\-'][a-z0-9]+)*", s.lower())
    out = []
    for t in toks:
        if t in _STOP or len(t) < 2:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def split_sentences(text: str) -> list[str]:
    text = _CITE.sub(lambda m: m.group(0) + " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n+", text)
    return [p.strip(" -*•\t") for p in parts if len(p.strip(" -*•\t")) > 3]


def parse_citations(text: str, k: int) -> list[int]:
    seen: list[int] = []
    for m in _CITE.finditer(text):
        for n in re.split(r"\s*[,;]\s*", m.group(1)):
            i = int(n)
            if 1 <= i <= k and i not in seen:
                seen.append(i)
    return seen



def normalize_citations(text: str) -> str:
    """Models cite in different styles: 【1】, [1†L3-L5], [Passage 1], [Source 2].
    Rewrite them all to [n] so the citation gate and the UI treat them the same."""
    text = re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", text)
    text = re.sub(r"\[(\d+)†[^\]]*\]", r"[\1]", text)
    text = re.sub(r"\[(?:passage|source|doc(?:ument)?)\s*#?(\d+)[^\]]*\]", r"[\1]", text, flags=re.I)
    return text

def is_refusal(text: str) -> bool:
    return bool(_IDK_RE.match(text.strip()))


def support_checks(answer: str, hits: list[Hit], threshold: float = 0.5) -> list[SentenceCheck]:
    """Lexical faithfulness proxy: for each cited sentence, what fraction of its
    content words appear in the passages it cites? Cheap, transparent, and
    good at catching numbers/names that came from nowhere."""
    checks = []
    for s in split_sentences(answer):
        cited = parse_citations(s, len(hits))
        if not cited:
            # Uncited sentence: judged against nothing, so support is 0 unless
            # it is pure connective text with no content words.
            words = _tokens(_CITE.sub("", s))
            if not words:
                continue
            checks.append(SentenceCheck(s, [], 0.0, False))
            continue
        evidence = set(_tokens(" ".join(hits[i - 1].chunk.text for i in cited)))
        words = _tokens(_CITE.sub("", s))
        if not words:
            continue
        frac = sum(w in evidence for w in words) / len(words)
        checks.append(SentenceCheck(s, cited, round(frac, 3), frac >= threshold))
    return checks


def format_passages(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] ({h.chunk.title} — {h.chunk.section})\n{h.chunk.text}")
    return "\n\n".join(blocks)


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 2,
                      min_overlap: float = 0.34) -> str:
    """No-LLM baseline: return the passage sentences that best match the question."""
    q = set(_tokens(question))
    if not q:
        return IDK
    scored = []
    for rank, h in enumerate(hits[:3], 1):
        for s in split_sentences(h.chunk.text):
            toks = set(_tokens(s))
            if len(toks) < 3:
                continue
            overlap = len(q & toks) / len(q)
            scored.append((overlap - 0.03 * (rank - 1), overlap, rank, s))
    scored.sort(reverse=True)
    if not scored or scored[0][1] < min_overlap:
        return IDK
    picked = []
    for _, overlap, rank, s in scored:
        if overlap < min_overlap or len(picked) >= max_sentences:
            break
        if all(s != p[1] for p in picked):
            picked.append((rank, s))
    return " ".join(f"{s.rstrip('.')}. [{rank}]" for rank, s in picked)


# ------------------------------------------------------------- pipeline ----
class GroundTruth:
    def __init__(self, cfg: Config, store: VectorStore | None = None, llm: BaseLLM | None = None,
                 rebuild: bool = False):
        self.cfg = cfg
        self.store = store or VectorStore.load_or_build(cfg, rebuild=rebuild)
        self.llm = llm or make_llm(cfg)
        self.threshold = cfg.refusal_threshold

    def ask(self, question: str, k: int | None = None) -> Answer:
        t0 = time.time()
        k = k or self.cfg.top_k
        hits = self.store.search(question, k)
        top = hits[0].score if hits else 0.0

        def done(text, refused, reason=None, raw=""):
            cited = [] if refused else parse_citations(text, len(hits))
            checks = [] if refused else support_checks(text, hits)
            return Answer(question, text, refused, reason, hits, cited, checks, raw,
                          round(time.time() - t0, 2), top)

        # GATE 1: retrieval confidence. Nothing relevant -> don't even ask the LLM.
        if not hits or top < self.threshold:
            return done(IDK, True, "low_retrieval")

        if self.llm.name == "extractive":
            text = extractive_answer(question, hits)
            if is_refusal(text):
                return done(IDK, True, "model_declined", text)
            return done(text, False, raw=text)

        user = f"Passages:\n\n{format_passages(hits)}\n\nQuestion: {question}"
        try:
            raw = self.llm.complete(SYSTEM_PROMPT, user)
            raw = normalize_citations(raw)
        except LLMError as e:
            return done(f"{IDK} (The language model could not be reached: {e})", True,
                        "llm_error", str(e))

        if is_refusal(raw):
            return done(IDK, True, "model_declined", raw)
        # GATE 2: an answer that cites nothing is not allowed through.
        if not parse_citations(raw, len(hits)):
            return done(IDK, True, "no_citation", raw)
        return done(raw, False, raw=raw)
