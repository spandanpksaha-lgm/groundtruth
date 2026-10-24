"""Convert a PDF (e.g. a student handbook) into clean Markdown for indexing.

    python scripts/pdf_to_markdown.py data/uc/Student_Handbook_2026_2027.pdf \
        data/uc/text/student-handbook.md --title "UC Student Handbook 2026-2027"

Why Markdown and not the PDF directly:
  * The PDF's bookmarks (its table of contents) become real headings, so the
    section-aware chunker can label every chunk "Section > Subsection".
  * A `<!-- page N -->` marker is kept at each page start, so answers can cite
    the page number.
  * The converted file is committed, so laptops that can't install PyMuPDF
    (some Intel Macs) still work.
"""
import argparse
import re
from pathlib import Path

import pymupdf


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def clean_lines(text: str) -> list[str]:
    lines = [l.rstrip() for l in text.splitlines()]
    # drop page-number-only lines and leading blanks
    lines = [l for l in lines if not re.fullmatch(r"\s*\d{1,3}\s*", l)]
    out: list[str] = []
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        # "•" on its own line, item text on the next line
        if l in {"•", "●", "▪", "◦", "o", "-", "–"} and i + 1 < len(lines):
            out.append("- " + lines[i + 1].strip())
            i += 2
            continue
        if l.startswith(("• ", "● ", "▪ ", "ï ")):
            l = "- " + l[2:].strip()
        out.append(l)
        i += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("out")
    p.add_argument("--title", required=True)
    p.add_argument("--source", help="path written to front matter (default: the PDF's file name)")
    p.add_argument("--max-heading-words", type=int, default=12)
    a = p.parse_args()

    doc = pymupdf.open(a.pdf)
    # Headings per page from the PDF's bookmarks; skip long "headings" that are
    # really sentences captured by the bookmark tool.
    toc: dict[int, list[tuple[int, str]]] = {}
    for level, title, page in doc.get_toc():
        title = title.strip()
        if (not title or len(title.split()) > a.max_heading_words or title.lower() == "table of contents"
                or title[0] in "•●▪-–" or title.endswith((":", "."))):
            continue
        toc.setdefault(page, []).append((level, title))

    toc_pages = {page for page in range(1, doc.page_count + 1)
                 if any(l == 1 and t.lower() == "table of contents" for l, t, pg in doc.get_toc() if pg == page)}

    parts = [f"---\ntitle: {a.title}\nsource: {a.source or Path(a.pdf).name}\n---\n"]
    skip_toc = False
    skipped: list[int] = []
    for pno, page in enumerate(doc, start=1):
        if pno in toc_pages:
            skip_toc = True
        lines = clean_lines(page.get_text("text"))
        # Skip the printed table of contents (pages full of dotted leaders).
        dotted = sum(1 for l in lines if re.search(r"\.{5,}\s*\d+$", l))
        if dotted >= 5 or (skip_toc and pno < 8 and dotted >= 2):
            continue
        # Skip pages that are mostly a table extracted one cell per line
        # (e.g. drug-schedule charts): as text they are noise for retrieval.
        words = [len(l.split()) for l in lines if l.strip()]
        if len(words) > 60 and sum(w <= 3 for w in words) / len(words) > 0.7:
            skipped.append(pno)
            continue
        heads = list(toc.get(pno, []))
        body: list[str] = []
        for l in lines:
            matched = None
            for k, (level, title) in enumerate(heads):
                if l and norm(l) == norm(title):
                    matched = k
                    break
            if matched is not None:
                level, title = heads.pop(matched)
                body += ["", "#" * (level + 1) + " " + title, ""]
            else:
                body.append(l)
        text = "\n".join(body).strip()
        if not text:
            continue
        marker = f"<!-- page {pno} -->"
        prev = parts[-1].rstrip() if len(parts) > 1 else ""
        if text.startswith(("#", "- ")) or (prev.endswith((".", ":", "!", "?")) and text[:1].isupper()):
            # new block: keep the paragraph break
            parts.append("\n" + marker + "\n" + text)
        else:
            # a sentence running across the page break stays whole
            parts.append(marker + text)
    md = "\n".join(parts)
    md = re.sub(r"\n{3,}", "\n\n", md)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md)
    print(f"Wrote {a.out}: {len(md.split())} words, {doc.page_count} pages, "
          f"{md.count(chr(10) + '#')} headings")
    if skipped:
        print(f"Skipped table-only pages (not indexed): {skipped}")


if __name__ == "__main__":
    main()
