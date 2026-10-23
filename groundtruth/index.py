"""Vector store: FAISS inner-product index over normalised embeddings
(= cosine similarity), saved to disk next to the chunk metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_corpus
from .config import Config
from .embeddings import Embedder, make_embedder
from .ingest import load_corpus

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - numpy fallback keeps things working
    faiss = None


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class VectorStore:
    def __init__(self, embedder: Embedder, chunks: list[Chunk], vectors: np.ndarray):
        self.embedder = embedder
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self._index = None
        if faiss is not None:
            self._index = faiss.IndexFlatIP(self.vectors.shape[1])
            self._index.add(self.vectors)

    # ---------------------------------------------------------- building --
    @classmethod
    def build(cls, cfg: Config, chunker: str | None = None, embedder: str | None = None,
              chunk_size: int | None = None, verbose: bool = True) -> "VectorStore":
        chunker = chunker or cfg.chunker
        size = chunk_size or cfg.chunk_size
        docs = load_corpus(cfg.data_dir, cfg.source_base_url)
        chunks = chunk_corpus(docs, chunker, size, cfg.chunk_overlap)
        emb = make_embedder(cfg, embedder)
        texts = [c.embed_text for c in chunks]
        emb.fit(texts)
        vecs = emb.embed(texts)
        if verbose:
            print(f"Indexed {len(docs)} documents -> {len(chunks)} chunks "
                  f"(chunker={chunker}, size={size}, embedder={emb.name}, dim={vecs.shape[1]})")
        return cls(emb, chunks, vecs)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with open(path / "chunks.jsonl", "w") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        self.embedder.save(path)
        (path / "meta.json").write_text(json.dumps({"embedder": self.embedder.name}))

    @classmethod
    def load(cls, cfg: Config, path: Path, embedder: str | None = None) -> "VectorStore":
        chunks = [Chunk(**json.loads(l)) for l in open(path / "chunks.jsonl")]
        vecs = np.load(path / "vectors.npy")
        emb = make_embedder(cfg, embedder).load(path)
        return cls(emb, chunks, vecs)

    @classmethod
    def load_or_build(cls, cfg: Config, chunker=None, embedder=None, chunk_size=None,
                      rebuild: bool = False, verbose: bool = True) -> "VectorStore":
        path = cfg.index_path(chunker, embedder, chunk_size)
        if not rebuild and (path / "chunks.jsonl").exists():
            return cls.load(cfg, path, embedder)
        store = cls.build(cfg, chunker, embedder, chunk_size, verbose)
        store.save(path)
        return store

    # --------------------------------------------------------- searching --
    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.embedder.embed([query])
        k = min(k, len(self.chunks))
        if self._index is not None:
            scores, idx = self._index.search(q, k)
            pairs = zip(idx[0].tolist(), scores[0].tolist())
        else:
            sims = (self.vectors @ q[0])
            top = np.argsort(-sims)[:k]
            pairs = ((int(i), float(sims[i])) for i in top)
        return [Hit(self.chunks[i], float(s), r) for r, (i, s) in enumerate(pairs) if i >= 0]
