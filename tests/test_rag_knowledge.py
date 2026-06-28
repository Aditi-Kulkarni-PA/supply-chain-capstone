"""
Smoke tests for RAG / ChromaDB vector store.

These tests verify:
1. The vectorstore directory and SLA knowledge file exist.
2. The persisted ChromaDB collection can be opened and is non-empty.
3. A keyword search over the raw chunks returns relevant SLA content.

Tests do NOT call the OpenAI API — they query ChromaDB directly.

Run from the project root:
    python -m pytest tests/test_rag_knowledge.py -v
"""

import sys
from pathlib import Path
import pytest
import chromadb

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR      = PROJECT_ROOT / "supply_chain_delivery_app"
VECTORSTORE  = APP_DIR / "vectorstore"
SLA_FILE     = APP_DIR / "knowledge" / "delivery_sla_github_ready.md"
COLLECTION   = "sla_knowledge"


# ── Pre-conditions ────────────────────────────────────────────────────────────

def test_sla_knowledge_file_exists():
    """SLA source document must be present for RAG to function."""
    assert SLA_FILE.exists(), f"SLA file not found at {SLA_FILE}"


def test_vectorstore_directory_exists():
    """Vectorstore directory must exist (created on first app launch)."""
    assert VECTORSTORE.exists(), (
        f"Vectorstore missing at {VECTORSTORE}. "
        "Launch the app once to build the collection."
    )


# ── ChromaDB collection ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def chroma_collection():
    """Open the persisted ChromaDB collection (read-only, no OpenAI calls)."""
    client = chromadb.PersistentClient(path=str(VECTORSTORE))
    try:
        coll = client.get_collection(name=COLLECTION)
    except Exception as exc:
        pytest.skip(f"Collection '{COLLECTION}' not yet built: {exc}")
    return coll


def test_collection_is_non_empty(chroma_collection):
    """Vector store must contain at least one indexed chunk."""
    count = chroma_collection.count()
    assert count > 0, "ChromaDB collection is empty — rebuild the vector store."


def test_collection_has_reasonable_chunk_count(chroma_collection):
    """Sanity-check: SLA doc should produce at least 5 chunks."""
    count = chroma_collection.count()
    assert count >= 5, f"Only {count} chunks found — SLA file may not have been fully indexed."


def test_peek_returns_documents(chroma_collection):
    """peek() should return actual document text, not empty strings."""
    result = chroma_collection.peek(limit=3)
    docs = result.get("documents", [])
    assert len(docs) > 0, "peek() returned no documents"
    for doc in docs:
        assert isinstance(doc, str) and len(doc) > 0, "Empty document chunk found"


def test_sla_chunks_contain_sla_keywords(chroma_collection):
    """At least one chunk should mention SLA/OLA/delivery terms."""
    result = chroma_collection.peek(limit=chroma_collection.count())
    docs = result.get("documents", [])
    sla_keywords = {"sla", "ola", "delivery", "delay", "partner", "target"}
    found = any(
        any(kw in doc.lower() for kw in sla_keywords)
        for doc in docs
    )
    assert found, "No SLA-related keywords found in any indexed chunk."


def test_where_document_filter(chroma_collection):
    """ChromaDB where_document filter should return SLA-related chunks."""
    # Use get() with a text filter — avoids embedding dimension mismatch
    # (collection was built with OpenAI 1536-dim embeddings)
    results = chroma_collection.get(
        where_document={"$contains": "SLA"},
        include=["documents"],
    )
    docs = results.get("documents", [])
    assert isinstance(docs, list) and len(docs) > 0, \
        "No chunks matched the SLA document filter"
