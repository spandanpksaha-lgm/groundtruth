"""Load a folder of documents (Markdown, PDF, HTML, TXT) into clean text.

Each document keeps its title and a source URL so every chunk can be cited
back to a page a human can open and check.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SKIP_NAMES = {"license", "license.md", "readme.md", "contributing.md"}


@dataclass
class Document:
    doc_id: str          # relative path, e.g. "travel-and-leave/leave.md"
    title: str
    url: str             # clickable source link
    text: str            # cleaned text; Markdown headings (#, ##, ...) preserved
    page_starts: list[tuple[int, int]] | None = None  # PDFs: (char_offset, page_no)


# ---------------------------------------------------------------- Markdown --
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_LIQUID_PAGE = re.compile(r"""\{%\s*page\s+["']([^"']+)["']\s*%\}""")
_LIQUID_ANY = re.compile(r"\{%.*?%\}", re.S)
_LIQUID_VAR = re.compile(r"\{\{.*?\}\}", re.S)
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_BLANK = re.compile(r"\n{3,}")


def _front_matter_title(fm: str) -> str | None:
    m = re.search(r"^title:\s*(.+)$", fm, re.M)
    if not m:
        return None
    return m.group(1).strip().strip('"').strip("'")


def clean_markdown(raw: str) -> tuple[str, str | None]:
    """Strip front matter, templating, link syntax and HTML; keep headings,
    lists and tables. Returns (text, title_from_front_matter)."""
    title = None
    m = _FRONT_MATTER.match(raw)
    if m:
        title = _front_matter_title(m.group(1))
        raw = raw[m.end():]
    raw = _LIQUID_PAGE.sub(r"\1", raw)
    raw = _LIQUID_ANY.sub("", raw)
    raw = _LIQUID_VAR.sub("", raw)
    raw = _MD_IMAGE.sub(r"\1", raw)
    raw = _MD_LINK.sub(r"\1", raw)
    raw = _HTML_TAG.sub("", raw)
    raw = raw.replace("**", "").replace("__", "")
    raw = re.sub(r"\\([\\`*_{}\[\]()#+\-.!$=&~|])", r"\1", raw)  # markdown escapes
    raw = _unwrap(raw)
    raw = _MULTI_BLANK.sub("\n\n", raw)
    return raw.strip(), title


_BLOCK_START = re.compile(r"^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|\||>)")


def _unwrap(text: str) -> str:
    """Join hard-wrapped lines back into sentences. Markdown sources wrap at
    ~80 columns; without this, sentence splitting and display break mid-line.
    Headings, list items, tables and quotes still start new lines."""
    out: list[str] = []
    for line in text.splitlines():
        s = line.rstrip()
        if (out and s.strip() and out[-1].strip() and not _BLOCK_START.match(s)
                and not out[-1].lstrip().startswith(("#", "|"))):
            out[-1] = out[-1] + " " + s.strip()
        else:
            out.append(s)
    return "\n".join(out)


# --------------------------------------------------------------------- PDF --
def load_pdf(path: Path) -> tuple[str, list[tuple[int, int]]]:
    import pymupdf  # imported lazily so Markdown-only users don't need it

    parts, page_starts, offset = [], [], 0
    with pymupdf.open(path) as pdf:
        for i, page in enumerate(pdf, start=1):
            t = page.get_text("text").strip()
            if not t:
                continue
            page_starts.append((offset, i))
            parts.append(t)
            offset += len(t) + 2
    return "\n\n".join(parts), page_starts


# -------------------------------------------------------------------- HTML --
def load_html(raw: str) -> tuple[str, str | None]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else None
    # Turn h1-h4 into Markdown headings so the section chunker can use them.
    for level in range(1, 5):
        for h in soup.find_all(f"h{level}"):
            h.replace_with(f"\n\n{'#' * level} {h.get_text(strip=True)}\n\n")
    text = soup.get_text("\n")
    text = _MULTI_BLANK.sub("\n\n", re.sub(r"[ \t]+\n", "\n", text))
    return text.strip(), title


# ------------------------------------------------------------------ loader --
def _url_for(rel: Path, base_url: str) -> str:
    if rel.suffix.lower() in {".md", ".markdown", ".html", ".htm"}:
        slug = rel.with_suffix("").as_posix()
        if slug.endswith("/index"):
            slug = slug[: -len("/index")]
        return f"{base_url.rstrip('/')}/{slug}/"
    return rel.as_posix()  # PDFs/TXT: the file itself is the source


def _title_from_text(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else fallback


def load_corpus(data_dir: Path, base_url: str = "") -> list[Document]:
    docs: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name.lower() in SKIP_NAMES:
            continue
        rel = path.relative_to(data_dir)
        ext = path.suffix.lower()
        page_starts = None
        if ext in {".md", ".markdown"}:
            text, title = clean_markdown(path.read_text(encoding="utf-8", errors="ignore"))
        elif ext in {".html", ".htm"}:
            text, title = load_html(path.read_text(encoding="utf-8", errors="ignore"))
        elif ext == ".pdf":
            text, page_starts = load_pdf(path)
            title = None
        elif ext == ".txt":
            text, title = path.read_text(encoding="utf-8", errors="ignore"), None
        else:
            continue
        if len(text.split()) < 20:  # skip empty/redirect stubs
            continue
        fallback = path.stem.replace("-", " ").replace("_", " ").title()
        title = _HTML_TAG.sub("", title or _title_from_text(text, fallback)).strip()
        docs.append(Document(
            doc_id=rel.as_posix(),
            title=title,
            url=_url_for(rel, base_url) if base_url else rel.as_posix(),
            text=text,
            page_starts=page_starts,
        ))
    return docs
