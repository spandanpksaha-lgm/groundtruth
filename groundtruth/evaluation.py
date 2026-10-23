"""Scoring. Every metric here is computed automatically and is reproducible;
the optional LLM judge adds a second opinion on correctness/faithfulness.

Per question we decide:
  retrieval_hit    all gold evidence strings appear in the top-k passages
  correct          answer contains a phrase from every key-fact group
  cites_evidence   the passages the answer CITES contain the gold evidence
  outcome          one label from the failure taxonomy below

Failure taxonomy (answerable questions)
  correct                 right answer, cites the right passage
  correct_bad_citation    right answer, but the cited passage lacks the evidence
  retrieval_miss          evidence never reached the top-k  -> retrieval problem
  false_refusal           evidence was retrieved but the system said "I don't know"
  wrong_answer            evidence retrieved, system answered, but got it wrong
Unanswerable questions
  correct_refusal         said "I don't know"
  hallucinated            answered something the documents don't support
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .rag import Answer, GroundTruth

STAKES_WEIGHT = {"high": 3, "medium": 2, "low": 1}


def load_questions(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("\\", "")
    return re.sub(r"\s+", " ", s).strip()


def _contains(haystack: str, phrase: str) -> bool:
    h, p = _norm(haystack), _norm(phrase)
    if re.fullmatch(r"[\d.,$%]+", p):           # bare numbers: whole-number match
        return re.search(rf"(?<![\d.]){re.escape(p)}(?![\d])", h) is not None
    return p in h


def key_facts_present(answer: str, key_facts: list[list[str]]) -> bool:
    return all(any(_contains(answer, alt) for alt in group) for group in key_facts)


def evidence_found(texts: list[str], evidence: list[str]) -> bool:
    blob = " ".join(texts)
    return all(_contains(blob, e) for e in evidence)


def evidence_rank(hits, evidence: list[str]) -> int | None:
    """1-based rank at which ALL evidence has appeared (cumulatively)."""
    for r in range(1, len(hits) + 1):
        if evidence_found([h.chunk.text for h in hits[:r]], evidence):
            return r
    return None


@dataclass
class Row:
    id: str
    category: str
    stakes: str
    answerable: bool
    question: str
    answer: str
    refused: bool
    refusal_reason: str | None
    top_score: float
    evidence_rank: int | None
    retrieval_hit: bool | None
    correct: bool | None
    cites_evidence: bool | None
    support_rate: float | None
    judge_correct: bool | None
    judge_faithful: float | None
    outcome: str
    retrieved: str
    cited: str
    latency_s: float


def score_one(q: dict, a: Answer, judge=None) -> Row:
    rank = evidence_rank(a.hits, q["evidence"]) if q["answerable"] else None
    hit = rank is not None if q["answerable"] else None
    correct = cites = None
    j_correct = j_faith = None

    if q["answerable"]:
        correct = (not a.refused) and key_facts_present(a.text, q["key_facts"])
        if not a.refused:
            cites = evidence_found([h.chunk.text for h in a.cited_hits], q["evidence"])
        if correct and cites:
            outcome = "correct"
        elif correct:
            outcome = "correct_bad_citation"
        elif not hit:
            outcome = "retrieval_miss"
        elif a.refused:
            outcome = "false_refusal"
        else:
            outcome = "wrong_answer"
    else:
        outcome = "correct_refusal" if a.refused else "hallucinated"

    if judge is not None and not a.refused:
        j_correct = judge.correct(q, a) if q["answerable"] else False
        j_faith = judge.faithful(a)

    return Row(
        id=q["id"], category=q["category"], stakes=q.get("stakes", "medium"),
        answerable=q["answerable"], question=q["question"], answer=a.text,
        refused=a.refused, refusal_reason=a.refusal_reason, top_score=round(a.top_score, 4),
        evidence_rank=rank, retrieval_hit=hit, correct=correct, cites_evidence=cites,
        support_rate=None if a.support_rate is None else round(a.support_rate, 3),
        judge_correct=j_correct, judge_faithful=j_faith, outcome=outcome,
        retrieved=" | ".join(f"{h.chunk.chunk_id}:{h.score:.3f}" for h in a.hits),
        cited=" | ".join(h.chunk.chunk_id for h in a.cited_hits),
        latency_s=a.latency_s,
    )


def _rate(num, den):
    return None if den == 0 else round(num / den, 3)


def summarise(rows: list[Row], k: int) -> dict:
    ans = [r for r in rows if r.answerable]
    una = [r for r in rows if not r.answerable]
    answered = [r for r in ans if not r.refused]
    supp = [r.support_rate for r in rows if r.support_rate is not None and r.answerable]
    s = {
        "n_questions": len(rows), "n_answerable": len(ans), "n_unanswerable": len(una),
        "answer_correctness": _rate(sum(r.correct for r in ans), len(ans)),
        "citation_faithfulness": _rate(sum(bool(r.cites_evidence) for r in answered), len(answered)),
        "sentence_support_rate": None if not supp else round(sum(supp) / len(supp), 3),
        "refusal_accuracy": _rate(sum(r.refused for r in una), len(una)),
        "false_refusal_rate": _rate(sum(r.refused for r in ans), len(ans)),
        f"retrieval_hit@{k}": _rate(sum(bool(r.retrieval_hit) for r in ans), len(ans)),
        "retrieval_hit@1": _rate(sum(r.evidence_rank == 1 for r in ans), len(ans)),
        "mrr": round(sum(1 / r.evidence_rank for r in ans if r.evidence_rank) / max(1, len(ans)), 3),
        "outcomes": {},
        "by_category": {},
        "harm_weighted_error": None,
        "avg_latency_s": round(sum(r.latency_s for r in rows) / max(1, len(rows)), 2),
    }
    for r in rows:
        s["outcomes"][r.outcome] = s["outcomes"].get(r.outcome, 0) + 1
    for cat in sorted({r.category for r in rows}):
        rs = [r for r in rows if r.category == cat]
        good = sum(r.outcome in ("correct", "correct_refusal") for r in rs)
        s["by_category"][cat] = {"n": len(rs), "fully_correct": _rate(good, len(rs))}
    # Harm-weighted error: errors on high-stakes questions count 3x low-stakes ones.
    bad = {"retrieval_miss", "false_refusal", "wrong_answer", "hallucinated"}
    w_err = sum(STAKES_WEIGHT[r.stakes] for r in rows if r.outcome in bad)
    w_all = sum(STAKES_WEIGHT[r.stakes] for r in rows)
    s["harm_weighted_error"] = _rate(w_err, w_all)
    if any(r.judge_correct is not None for r in rows):
        jc = [r for r in ans if r.judge_correct is not None]
        jf = [r.judge_faithful for r in rows if r.judge_faithful is not None]
        s["judge_answer_correctness"] = _rate(sum(r.judge_correct for r in jc), len(ans))
        s["judge_faithfulness"] = None if not jf else round(sum(jf) / len(jf), 3)
    return s


def run(gt: GroundTruth, questions: list[dict], k: int, judge=None, verbose=True) -> tuple[list[Row], dict]:
    rows = []
    for q in questions:
        a = gt.ask(q["question"], k=k)
        row = score_one(q, a, judge)
        rows.append(row)
        if verbose:
            mark = "✓" if row.outcome in ("correct", "correct_refusal") else "✗"
            print(f"  {mark} {q['id']:4} {row.outcome:22} top={row.top_score:.3f}  {q['question'][:60]}")
    return rows, summarise(rows, k)


def save(rows: list[Row], summary: dict, out_dir: Path, config: dict) -> None:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_question.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    (out_dir / "summary.json").write_text(json.dumps({"config": config, **summary}, indent=2))


# ------------------------------------------------------------- LLM judge --
class LLMJudge:
    """Optional second opinion using the same LLM backend. Use a different
    (ideally stronger) model than the generator if you can, so the system
    isn't grading its own homework."""

    def __init__(self, llm):
        self.llm = llm

    def correct(self, q: dict, a: Answer) -> bool:
        out = self.llm.complete(
            "You grade answers. Reply with exactly one word: CORRECT or INCORRECT.",
            f"Question: {q['question']}\nReference answer: {q['reference']}\n"
            f"Candidate answer: {a.text}\n\nIs the candidate answer factually consistent "
            f"with the reference and does it answer the question?", max_tokens=5)
        return out.strip().upper().startswith("CORRECT")

    def faithful(self, a: Answer) -> float | None:
        from .rag import parse_citations, split_sentences
        verdicts = []
        for s in split_sentences(a.text):
            cited = parse_citations(s, len(a.hits))
            if not cited:
                continue
            ev = "\n\n".join(a.hits[i - 1].chunk.text for i in cited)
            out = self.llm.complete(
                "You check citations. Reply with exactly one word: SUPPORTED or UNSUPPORTED.",
                f"Passage(s):\n{ev}\n\nClaim: {s}\n\nIs every fact in the claim stated in the passage(s)?",
                max_tokens=5)
            verdicts.append(out.strip().upper().startswith("SUPPORTED"))
        return None if not verdicts else round(sum(verdicts) / len(verdicts), 3)
