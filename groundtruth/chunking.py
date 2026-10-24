"""Two chunking strategies, so we can compare them with numbers.

* ``fixed``   - naive baseline: a sliding window of N words with overlap,
                blind to document structure.
* ``section`` - structure-aware: splits on Markdown headings, packs small
                sections together, splits long sections on paragraph
                boundaries, and prefixes every chunk with its
                "Title > Section > Subsection" path so the embedding knows
                where the text came from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .ingest import Document

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str        # "<doc_id>#<n>"
    doc_id: str
    title: str
    url: str
    section: str         # "Annual leave > Carrying over leave"
    text: str            # what the user sees / the LLM reads
    embed_text: str      # what gets embedded (may include the section path)
    page: int | None = None

    def citation_label(self) -> str:
        label = self.title
        if self.section and self.section != self.title:
            label += f" — {self.section}"
        if self.page:
            label += f" (p.{self.page})"
        return label

    def to_dict(self) -> dict:
        return asdict(self)


def _words(s: str) -> int:
    return len(s.split())


def _page_for(doc: Document, offset: int) -> int | None:
    if not doc.page_starts:
        return None
    page = doc.page_starts[0][1]
    for start, p in doc.page_starts:
        if start <= offset:
            page = p
        else:
            break
    return page


def _strip_heading_marks(text: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", text, flags=re.M)


# ----------------------------------------------------------------- fixed --
def chunk_fixed(doc: Document, size: int = 220, overlap: int = 40) -> list[Chunk]:
    text = _strip_heading_marks(doc.text)
    # Track heading at each word position so citations still show a section.
    words, sections, cur = [], [], ""
    for line in doc.text.splitlines():
        m = _HEADING.match(line)
        if m:
            cur = m.group(2).strip()
        for w in _strip_heading_marks(line).split():
            words.append(w)
            sections.append(cur)
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    cursor = 0
    for n, start in enumerate(range(0, max(1, len(words)), step)):
        piece = words[start:start + size]
        if not piece:
            break
        body = " ".join(piece)
        off = text.find(piece[0], cursor)
        cursor = off if off >= 0 else cursor
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}#{n}", doc_id=doc.doc_id, title=doc.title,
            url=doc.url, section=sections[start] or doc.title, text=body,
            embed_text=body, page=_page_for(doc, cursor),
        ))
        if start + size >= len(words):
            break
    return chunks


# --------------------------------------------------------------- section --
@dataclass
class _Section:
    path: list[str]
    body: str
    offset: int


def _split_sections(doc: Document) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    offset = 0
    buf_offset = 0

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append(_Section([h for _, h in stack], body, buf_offset))

    for line in doc.text.splitlines(keepends=True):
        m = _HEADING.match(line.rstrip("\n"))
        if m:
            flush()
            buf = []
            level, heading = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            buf_offset = offset
        else:
            if not buf:
                buf_offset = offset
            buf.append(line.rstrip("\n"))
        offset += len(line)
    flush()
    return sections


def _split_long(body: str, size: int, overlap: int) -> list[str]:
    """Split on blank lines (paragraphs/lists), packing up to `size` words.
    A single paragraph longer than `size` is split by words."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    cur: list[str] = []
    for p in paras:
        if _words(p) > size:
            if cur:
                pieces.append("\n\n".join(cur))
                cur = []
            w = p.split()
            step = max(1, size - overlap)
            for i in range(0, len(w), step):
                pieces.append(" ".join(w[i:i + size]))
                if i + size >= len(w):
                    break
            continue
        if cur and _words("\n\n".join(cur + [p])) > size:
            pieces.append("\n\n".join(cur))
            # carry the last paragraph forward as overlap if it is short
            cur = [cur[-1]] if _words(cur[-1]) <= overlap else []
        cur.append(p)
    if cur:
        pieces.append("\n\n".join(cur))
    return pieces


def chunk_section(doc: Document, size: int = 220, overlap: int = 40,
                  min_words: int = 40) -> list[Chunk]:
    sections = _split_sections(doc)
    units: list[tuple[str, str, int]] = []  # (section_path, text, offset)
    for s in sections:
        path = " > ".join(s.path) if s.path else doc.title
        heading_line = s.path[-1] if s.path else ""
        if _words(s.body) > size:
            cursor = s.offset
            for piece in _split_long(s.body, size, overlap):
                found = doc.text.find(piece[:60], cursor)
                if found >= 0:
                    cursor = found
                units.append((path, piece, cursor))
        else:
            text = f"{heading_line}\n{s.body}".strip() if heading_line else s.body
            units.append((path, text, s.offset))

    # Pack tiny neighbouring units so we don't index 10-word fragments.
    packed: list[tuple[str, str, int]] = []
    for path, text, off in units:
        if packed:
            p_path, p_text, p_off = packed[-1]
            if (_words(p_text) < min_words or _words(text) < min_words) and \
               _words(p_text) + _words(text) <= size:
                # cite the page where most of the text is
                off2 = off if _words(p_text) < _words(text) else p_off
                packed[-1] = (p_path, f"{p_text}\n\n{text}", off2)
                continue
        packed.append((path, text, off))

    chunks = []
    for n, (path, text, off) in enumerate(packed):
        header = f"{doc.title} > {path}" if not path.startswith(doc.title) else path
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}#{n}", doc_id=doc.doc_id, title=doc.title,
            url=doc.url, section=path, text=text,
            embed_text=f"{header}\n{text}", page=_page_for(doc, off),
        ))
    return chunks


def chunk_corpus(docs: list[Document], strategy: str = "section",
                 size: int = 220, overlap: int = 40) -> list[Chunk]:
    fn = {"fixed": chunk_fixed, "section": chunk_section}.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown chunker '{strategy}'. Use 'fixed' or 'section'.")
    out: list[Chunk] = []
    for d in docs:
        out.extend(fn(d, size=size, overlap=overlap))
    return out
