"""Retrieval and rationale generation.

Sentence-BERT embeddings indexed with FAISS, with a TF-IDF keyword retriever
as a fallback. The fallback is not just an offline convenience -- it is the
comparison baseline. "Retrieval improved 20% over keyword search" is only a
claim you can make if the keyword retriever actually exists and is measured on
the same queries, so both live here behind one interface.

Explanations are template-composed from retrieved passages, never generated
free-form. A model asked to write a clinical rationale will produce a fluent
citation for a claim the passage does not support; the template can only
restate what was retrieved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RAGConfig
from ..types import TrackFeatures
from .corpus import Passage


@dataclass(frozen=True)
class RetrievedPassage:
    passage: Passage
    score: float
    rank: int


class KeywordRetriever:
    """TF-IDF cosine similarity. The baseline, and the offline fallback."""

    name = "tfidf"

    def __init__(self, passages: tuple[Passage, ...]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not passages:
            raise ValueError("cannot build a retriever over an empty corpus")
        self.passages = passages
        self._vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        corpus = [f"{p.text} {' '.join(p.tags)}" for p in passages]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int) -> tuple[RetrievedPassage, ...]:
        from sklearn.metrics.pairwise import cosine_similarity

        if not query.strip():
            raise ValueError("query must not be empty")
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix).ravel()
        order = np.argsort(-scores)[:top_k]
        return tuple(
            RetrievedPassage(self.passages[i], float(scores[i]), rank)
            for rank, i in enumerate(order)
        )


class SemanticRetriever:
    """Sentence-BERT + FAISS inner-product over L2-normalised embeddings."""

    name = "sbert_faiss"

    def __init__(self, passages: tuple[Passage, ...], cfg: RAGConfig) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        if not passages:
            raise ValueError("cannot build a retriever over an empty corpus")
        self.passages = passages
        self.cfg = cfg
        self._model = SentenceTransformer(cfg.embedding_model)

        embeddings = self._encode([f"{p.text} {' '.join(p.tags)}" for p in passages])
        if embeddings.shape[1] != cfg.embedding_dim:
            raise ValueError(
                f"model {cfg.embedding_model!r} produced {embeddings.shape[1]}-dim vectors, "
                f"config expects {cfg.embedding_dim}"
            )
        # Inner product on normalised vectors == cosine similarity.
        self._index = faiss.IndexFlatIP(cfg.embedding_dim)
        self._index.add(embeddings)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms < 1e-12, 1.0, norms)

    def search(self, query: str, top_k: int) -> tuple[RetrievedPassage, ...]:
        if not query.strip():
            raise ValueError("query must not be empty")
        scores, indices = self._index.search(self._encode([query]), min(top_k, len(self.passages)))
        return tuple(
            RetrievedPassage(self.passages[int(idx)], float(score), rank)
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]))
            if idx >= 0
        )


def build_retriever(passages: tuple[Passage, ...], cfg: RAGConfig):
    """Semantic retriever when available, keyword retriever otherwise.

    Falls back rather than failing: the first run downloads a model, and a
    machine without network access should still be able to run the pipeline.
    The caller can see which one it got from `.name`.
    """
    if cfg.allow_download:
        try:
            return SemanticRetriever(passages, cfg)
        except (ImportError, OSError, ValueError) as exc:
            if not cfg.fallback_to_tfidf:
                raise
            print(f"[rag] semantic retriever unavailable ({type(exc).__name__}: {exc}); "
                  "falling back to TF-IDF keyword retrieval")
    return KeywordRetriever(passages)


def build_query(track: TrackFeatures, stress: float) -> str:
    """Turn a recommendation into a retrieval query in the corpus's vocabulary."""
    descriptors = [
        "high stress arousal reduction" if stress >= 7 else "moderate stress relaxation",
        f"{'slow' if track.tempo_bpm < 80 else 'moderate' if track.tempo_bpm < 110 else 'fast'} tempo",
        "sustained drone" if track.drone else "no drone",
        f"{'sparse' if track.rhythmic_intensity < 0.4 else 'dense'} rhythmic texture",
        "alpha power beta alpha ratio frontal EEG",
        f"raga {track.raga}",
    ]
    return ", ".join(descriptors)


def explain(
    track: TrackFeatures,
    stress: float,
    retriever,
    cfg: RAGConfig,
    predicted_delta_alpha: float | None = None,
) -> str:
    """Compose an evidence-grounded rationale from retrieved passages."""
    retrieved = retriever.search(build_query(track, stress), cfg.top_k)
    if not retrieved:
        raise RuntimeError("retriever returned no passages; the index may be empty")

    properties = [f"tempo {track.tempo_bpm:.0f} BPM"]
    if track.drone:
        properties.append("sustained drone")
    properties.append(f"rhythmic intensity {track.rhythmic_intensity:.2f}")

    lines = [
        f"Recommended: {track.raga} ({track.track_id}) -- {', '.join(properties)}.",
        f"Detected stress level: {stress:.1f}/10.",
    ]
    if predicted_delta_alpha is not None:
        lines.append(
            f"Predicted alpha change: {predicted_delta_alpha:+.2f} uV^2 "
            "(model estimate, not a measurement)."
        )
    lines.append("")
    lines.append("Grounding:")
    for item in retrieved:
        flag = "" if item.passage.verified else "  [UNVERIFIED CITATION]"
        lines.append(f"  [{item.rank + 1}] {item.passage.text}")
        lines.append(f"      -- {item.passage.citation} (similarity {item.score:.3f}){flag}")

    unverified = sum(1 for item in retrieved if not item.passage.verified)
    if unverified:
        lines.append("")
        lines.append(
            f"WARNING: {unverified} of {len(retrieved)} retrieved passages have "
            "unverified citations. Not suitable for clinical use as-is."
        )
    return "\n".join(lines)


def compare_retrievers(
    semantic, keyword, queries: list[str], top_k: int
) -> dict[str, float]:
    """Overlap between the two retrievers' top-k, per query.

    This measures *divergence*, not accuracy -- without relevance judgements
    there is no ground truth to score against. To make a "+20% retrieval
    accuracy" claim you need human-labelled query/passage relevance; this
    function deliberately does not pretend to supply it.
    """
    if not queries:
        raise ValueError("need at least one query")
    overlaps = []
    for query in queries:
        a = {r.passage.passage_id for r in semantic.search(query, top_k)}
        b = {r.passage.passage_id for r in keyword.search(query, top_k)}
        overlaps.append(len(a & b) / max(len(a | b), 1))
    return {
        "mean_jaccard_overlap": float(np.mean(overlaps)),
        "n_queries": float(len(queries)),
    }
