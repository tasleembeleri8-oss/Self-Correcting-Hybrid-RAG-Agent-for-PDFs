"""Retrieval. Every function takes a doc_id and searches ONLY that document's indexes."""
from .documents import get_document, tokenize


class DocumentNotFound(Exception):
    """The document id is unknown, or the document has expired."""


def _load(doc_id: str):
    doc = get_document(doc_id)
    if doc is None:
        raise DocumentNotFound(doc_id)
    return doc


def vector_search(doc_id: str, query: str, k: int = 5) -> list:
    """Semantic search only: finds chunks whose MEANING is close to the query."""
    doc = _load(doc_id)
    result = doc.collection.query(query_texts=[query], n_results=min(k, len(doc.chunks)))
    return result["documents"][0]


def hybrid_search(doc_id: str, query: str, k: int = 5) -> list:
    """Vector search + BM25 keyword search, merged with Reciprocal Rank Fusion (RRF).

    Vector search is good at meaning ("natural sweetener" ~ "xylitol") but can miss exact
    terms (codes, names, numbers). BM25 is the opposite. Using both covers both weaknesses.
    """
    doc = _load(doc_id)
    n = len(doc.chunks)
    pool = min(10, n)                       # candidates taken from each search method

    # 1) vector candidates, best first
    vec_ids = doc.collection.query(query_texts=[query], n_results=pool)["ids"][0]

    # 2) BM25 candidates, best first. Chunks with score 0 share no words with the query, so
    #    they are noise and are skipped.
    scores = doc.bm25.get_scores(tokenize(query))
    ranked = sorted(range(n), key=lambda i: -scores[i])[:pool]
    bm_ids = [f"c{i}" for i in ranked if scores[i] > 0]

    # 3) RRF: score = sum of 1 / (60 + rank) over each list the chunk appears in.
    #    It merges two ranked lists using only RANKS, so we never have to compare a vector
    #    distance with a BM25 score (those numbers are on different scales).
    fused = {}
    for ranking in (vec_ids, bm_ids):
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)

    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [doc.chunks[int(cid[1:])][0] for cid in top]
