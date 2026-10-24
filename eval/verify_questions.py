"""Check a question file against its corpus: every evidence string must
appear in the documents, and every 'unanswerable' question must not have its
key terms answered. Run after writing or editing questions.

    python eval/verify_questions.py --corpus uc
    python eval/verify_questions.py --corpus uc --split hard
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from groundtruth.config import get_config  # noqa: E402
from groundtruth.evaluation import _norm  # noqa: E402
from groundtruth.ingest import load_corpus  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=["tts", "uc"], default="tts")
    p.add_argument("--split", nargs="+", default=["test", "dev"])
    a = p.parse_args(argv)
    cfg = get_config(corpus=a.corpus)
    text = _norm(" ".join(d.text for d in load_corpus(cfg.data_dir, cfg.source_base_url)))
    bad = 0
    for split in a.split:
        path = cfg.questions_path(split)
        if not path.exists():
            print(f"(no {path.name})")
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        ids = [r["id"] for r in rows]
        if len(ids) != len(set(ids)):
            print(f"DUPLICATE IDS in {path.name}")
            bad += 1
        for r in rows:
            if r["answerable"]:
                if not r.get("evidence") or not r.get("key_facts"):
                    print(f"  {r['id']}: answerable but missing evidence or key_facts")
                    bad += 1
                for e in r.get("evidence", []):
                    if _norm(e) not in text:
                        print(f"  {r['id']}: evidence not found in corpus: {e!r}")
                        bad += 1
        print(f"{path.name}: {len(rows)} questions, {sum(not r['answerable'] for r in rows)} unanswerable")
    print("OK" if not bad else f"{bad} problem(s)")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
