"""Bonus A: chunking comparison. Bonus B: cross-encoder reranking.

Run with the same Python environment and OpenAI key as starter_code.py:
    python bonus.py --check       # Local chunking check; no API requests
    python bonus.py --part a      # Compare top-3 retrieval for three queries
    python bonus.py               # Both bonuses, including before/after answers

Bonus B also requires, for this assignment's NumPy 1.26 environment:
    python -m pip install "sentence-transformers==3.4.1" "scipy==1.15.3"
Its first run downloads cross-encoder/ms-marco-MiniLM-L6-v2.
The report is saved as ../bonus_results.md. Chroma indexes are reused on reruns.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Reuse the assignment's document loader, questions, and API-key configuration.
from starter_code import TEST_QUERIES, load_documents

BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CHUNKING_QUERIES = (5, 7, 10)
RERANKING_QUERIES = (5, 10)
FINAL_K = 3

# Hand-specified evidence checks grounded in the provided synthetic KB.
# These are transparent retrieval diagnostics, not precision/recall or a complete
# answer-correctness metric. A fact must appear in the expected source document.
EVIDENCE = {
    5: [
        ("Initial announcement and continuing updates", "IROPS_Procedures.txt",
         ("within 15 minutes", "every 30 minutes")),
        ("Special procedure for delays over two hours", "IROPS_Procedures.txt",
         ("delays exceeding 2 hours", "directly notify connecting passengers")),
        ("Passenger Connect identifies connecting gates", "IROPS_Procedures.txt",
         ("Passenger Connect", "gate locations")),
        ("Personal gate visit and receiving-agent coordination", "IROPS_Procedures.txt",
         ("personally approach each connecting gate", "receiving agent")),
    ],
    7: [
        ("Weather rebooking priority by status", "Customer_Rebooking_Policy.txt",
         ("Priority 1: Platinum", "Priority 4: Standard")),
        ("All four status-tier rebooking windows", "Customer_Rebooking_Policy.txt",
         ("24 hours from original departure", "48 hours from original departure",
          "72 hours from original departure", "7 days from original departure")),
        ("Gold partner-airline condition", "Customer_Rebooking_Policy.txt",
         ("partner airline metal", "within 4 hours")),
        ("Platinum any-carrier condition", "Customer_Rebooking_Policy.txt",
         ("any carrier", "airline or partner", "within 3 hours")),
        ("Weather hotel exceptions by tier", "Customer_Rebooking_Policy.txt",
         ("NOT provided to Standard or Silver", "retain their hotel")),
        ("Weather fee AND fare-difference waiver for all tiers", "Customer_Rebooking_Policy.txt",
         ("all change fees and fare differences", "regardless of tier", "within 14 days")),
    ],
    10: [
        ("FDP limits constrain recovery despite aircraft readiness", "Crew_Rest_Regulations.txt",
         ("maximum FDP", "even if the aircraft is ready")),
        ("Local and regional reserve-crew recovery", "Crew_Rest_Regulations.txt",
         ("Reserve crew must be activated", "regional crew bases")),
        ("Earliest legal crew time communicated to OCC", "Crew_Rest_Regulations.txt",
         ("Earliest time any available crew can be legal", "OCC Duty Manager")),
        ("Cascading recovery can take 24–36 hours", "Crew_Rest_Regulations.txt",
         ("24–36 hours",)),
        ("OCC coordinates aircraft, crew, passenger, and gate recovery", "IROPS_Procedures.txt",
         ("OCC Recovery Team", "Crew recovery pairings", "Gate and ground resource allocation")),
        ("Two-hour recovery reviews and station confirmation", "IROPS_Procedures.txt",
         ("reviewed every 2 hours", "confirm receipt")),
    ],
}


def split_documents(docs, strategy):
    """Use the two requested configurations, both measured in CHARACTERS."""
    if strategy == "fixed":
        splitter = CharacterTextSplitter(
            chunk_size=500, chunk_overlap=0, add_start_index=True
        )
    elif strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=150, add_start_index=True
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    chunks = splitter.split_documents(docs)
    for index, chunk in enumerate(chunks, 1):
        chunk.metadata.update(strategy=strategy, chunk_id=f"{strategy}-{index:03d}")
    return chunks


def build_or_load_store(chunks, strategy, embeddings, cache_dir):
    """Cache by content/configuration, keeping the starter's index untouched."""
    payload = {
        "embedding_model": EMBEDDING_MODEL,
        "distance": "cosine",
        "chunks": [(d.page_content, d.metadata) for d in chunks],
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cache_dir) / f"{strategy}_{fingerprint}"
    store = Chroma(
        collection_name="airline_bonus",
        persist_directory=str(directory),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    existing_ids = set(store.get()["ids"])
    missing = [d for d in chunks if d.metadata["chunk_id"] not in existing_ids]
    if missing:
        print(f"Embedding {len(missing)} {strategy} chunks...", flush=True)
        store.add_documents(missing, ids=[d.metadata["chunk_id"] for d in missing])
    else:
        print(f"Reusing {len(chunks)} {strategy} chunks from {directory}", flush=True)
    return store


def retrieve(store, query_vector, k):
    """Chroma returns a distance: lower values are better."""
    return [
        {"doc": doc, "distance": float(distance), "initial_rank": rank}
        for rank, (doc, distance) in enumerate(
            store.similarity_search_by_vector_with_relevance_scores(query_vector, k=k), 1
        )
    ]


def normalize(text):
    return re.sub(r"\s+", " ", text).casefold()


def evidence_coverage(number, hits):
    by_source = {}
    for hit in hits:
        doc = hit["doc"]
        source = Path(doc.metadata["source"]).name
        by_source[source] = by_source.get(source, "") + "\n" + doc.page_content
    by_source = {name: normalize(text) for name, text in by_source.items()}
    found, missing = [], []
    for label, source, phrases in EVIDENCE[number]:
        context = by_source.get(source, "")
        supported = all(normalize(phrase) in context for phrase in phrases)
        (found if supported else missing).append(label)
    return found, missing


def format_hits(hits):
    """Show complete chunks, with source and position for inspection."""
    sections = []
    for rank, hit in enumerate(hits, 1):
        doc = hit["doc"]
        score = f"cosine distance={hit['distance']:.4f} (lower is better)"
        if "rerank_score" in hit:
            score += f"; cross-encoder score={hit['rerank_score']:.4f} (higher is better)"
        sections.append(
            f"#### Rank {rank}: {doc.metadata['chunk_id']}\n\n"
            f"Source: `{doc.metadata['source']}`; start character: "
            f"{doc.metadata['start_index']}; length: {len(doc.page_content)} characters; "
            f"initial vector rank: {hit['initial_rank']}; {score}.\n\n"
            f"```text\n{doc.page_content}\n```"
        )
    return "\n\n".join(sections)


def coverage_text(number, hits):
    found, missing = evidence_coverage(number, hits)
    return (
        f"Evidence checklist: {len(found)}/{len(EVIDENCE[number])}. "
        f"Found: {'; '.join(found) or 'none'}. "
        f"Not found: {'; '.join(missing) or 'none'}."
    )


def compare_chunking(stores, vectors):
    report = ["# Bonus A — Chunking Strategy Comparison"]
    totals = {strategy: 0 for strategy in stores}
    for number in CHUNKING_QUERIES:
        report.append(f"## Query {number}\n\n{TEST_QUERIES[number - 1]}")
        results = {}
        for strategy, store in stores.items():
            hits = retrieve(store, vectors[number], k=3)
            results[strategy] = evidence_coverage(number, hits)
            totals[strategy] += len(results[strategy][0])
            report.extend([
                f"### {strategy.title()} — top 3",
                format_hits(hits),
                coverage_text(number, hits),
            ])
        fixed, recursive = (len(results[s][0]) for s in ("fixed", "recursive"))
        if fixed == recursive:
            assessment = "Both strategies cover the same number of checklist items."
        else:
            winner = "Recursive" if recursive > fixed else "Fixed"
            assessment = f"{winner} retrieves more of the required evidence within the top three."
        fixed_only = [label for label in results["fixed"][0] if label not in results["recursive"][0]]
        recursive_only = [label for label in results["recursive"][0] if label not in results["fixed"][0]]
        both_missing = [label for label in results["fixed"][1] if label in results["recursive"][1]]
        report.append(
            f"### Qualitative assessment\n\n{assessment} "
            f"Fixed uniquely covers: {'; '.join(fixed_only) or 'no additional checklist items'}. "
            f"Recursive uniquely covers: {'; '.join(recursive_only) or 'no additional checklist items'}. "
            f"Both miss: {'; '.join(both_missing) or 'no checklist items'}. "
            "The complete chunks also show how much surrounding policy context is preserved."
        )
    best = max(totals, key=totals.get)
    recommendation = (
        f"On these three queries, {best} has stronger top-three checklist coverage "
        f"(fixed={totals['fixed']}, recursive={totals['recursive']}). "
        f"I recommend {best} for this tested configuration."
        if totals["fixed"] != totals["recursive"]
        else "The strategies tie on total checklist coverage; this sample does not establish "
        "a winner. I would retain recursive splitting as a provisional default for its "
        "overlap and fallback boundaries, pending evaluation on more queries."
    )
    report.append(
        "## Recommendation\n\n" + recommendation + " Recursive splitting provides overlap "
        "and fallback boundaries, while smaller fixed chunks may focus retrieval but separate "
        "related rules. Validate the choice on more queries, especially policy exceptions and "
        "cross-document synthesis. This comparison changes size, overlap, and splitting method "
        "together, so it does not isolate the effect of overlap alone."
    )
    return "\n\n".join(report)


def load_reranker(model_name):
    # API/model references:
    # https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html
    # https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Bonus B requires sentence-transformers. Install it in this Python "
            "environment, or use --part a to run only Bonus A."
        ) from exc
    return CrossEncoder(model_name, max_length=512, device="cpu")


def rerank(query, candidates, model):
    """Score the SAME eight candidates; no new retrieval after reranking."""
    scores = model.predict(
        [(query, hit["doc"].page_content) for hit in candidates], show_progress_bar=False
    )
    if len(scores) != len(candidates):
        raise ValueError("Reranker returned an unexpected number of scores")
    scored = [dict(hit, rerank_score=float(score)) for hit, score in zip(candidates, scores)]
    return sorted(scored, key=lambda hit: hit["rerank_score"], reverse=True)


def generate_answer(llm, query, hits):
    """Use the same context-only, source-cited approach as starter_code.py."""
    context = "\n\n".join(
        f"Content: {hit['doc'].page_content}\nSource: {hit['doc'].metadata['source']}"
        for hit in hits
    )
    prompt = (
        "You are an expert airline operations assistant.\n"
        "Use only the context below to answer the question.\n"
        "At the end, cite the source document(s) on a separate line beginning with SOURCES:.\n"
        "If the answer is not in the context, say: "
        "This information is not available in the operational documents.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (with source citation):"
    )
    return llm.invoke(prompt).content


def discuss_answers(llm, number, before, after, docs):
    """A labeled qualitative judge uses full reference documents, not scores."""
    sources = {source for _, source, _ in EVIDENCE[number]}
    reference = "\n\n".join(
        f"SOURCE: {doc.metadata['source']}\n{doc.page_content}"
        for doc in docs if Path(doc.metadata["source"]).name in sources
    )
    return llm.invoke(
        "Compare these two RAG answers against the supplied synthetic operational documents. "
        "In one concise paragraph, identify concrete correct, missing, or unsupported details "
        "and source-citation issues in each answer. State whether AFTER is better, unchanged, "
        "or worse, and why. Do not assume reranking improves quality. For Query 10, distinguish "
        "citing two files from actually synthesizing crew constraints with OCC recovery actions. "
        "Judge only from the reference text, not real-world regulations.\n\n"
        f"QUESTION {number}: {TEST_QUERIES[number - 1]}\n\n"
        f"REFERENCE DOCUMENTS:\n{reference}\n\nBEFORE:\n{before}\n\nAFTER:\n{after}"
    ).content


def compare_reranking(store, vectors, reranker, llm, docs):
    report = [
        "# Bonus B — Cross-Encoder Reranking",
        "Retrieve k=8 from the recursive index. BEFORE uses vector ranks 1–3; AFTER "
        "uses the cross-encoder's top 3 from the same pool. Both answers use the same "
        "model, temperature=0, prompt, and context count. Distances and reranker scores "
        "are different scales. Pairs longer than the model's 512-token limit are truncated. "
        "The answer comparison is LLM-assisted qualitative feedback, not a correctness guarantee.",
    ]
    for number in RERANKING_QUERIES:
        query = TEST_QUERIES[number - 1]
        candidates = retrieve(store, vectors[number], k=8)
        reordered = rerank(query, candidates, reranker)
        before_hits, after_hits = candidates[:FINAL_K], reordered[:FINAL_K]
        before = generate_answer(llm, query, before_hits)
        after = generate_answer(llm, query, after_hits)
        print(f"Completed reranking and answers for Query {number}.", flush=True)
        report.extend([
            f"## Query {number}\n\n{query}",
            "### Before reranking — all 8 candidates\n\n" + format_hits(candidates),
            "Baseline answer uses the first 3 chunks.\n\n" + coverage_text(number, before_hits),
            "### After reranking — top 3\n\n" + format_hits(after_hits),
            coverage_text(number, after_hits),
            "### Answer before reranking\n\n" + before,
            "### Answer after reranking\n\n" + after,
            "### Impact on answer quality (LLM-assisted)\n\n"
            + discuss_answers(llm, number, before, after, docs),
        ])
    return "\n\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--part", choices=("a", "b", "all"), default="all")
    parser.add_argument("--check", action="store_true", help="Check local chunks without API calls")
    parser.add_argument("--kb-dir", type=Path, default=BASE_DIR / "airline_kb")
    parser.add_argument("--cache-dir", type=Path, default=BASE_DIR / "bonus_chroma")
    parser.add_argument("--output", type=Path, default=BASE_DIR.parent / "bonus_results.md")
    parser.add_argument("--reranker-model", default=RERANKER_MODEL)
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    args = parser.parse_args()

    docs = load_documents(str(args.kb_dir))
    if not docs:
        raise ValueError(f"No knowledge-base documents found in {args.kb_dir}")
    docs.sort(key=lambda doc: doc.metadata["source"])
    for doc in docs:
        doc.metadata["source"] = f"airline_kb/{Path(doc.metadata['source']).name}"
    strategies = ("fixed", "recursive") if args.part != "b" else ("recursive",)
    chunks = {strategy: split_documents(docs, strategy) for strategy in strategies}
    settings = [
        "# Airline RAG Bonus Results",
        f"Embeddings: `{EMBEDDING_MODEL}`. Corpus: {len(docs)} documents. "
        "Bonus A uses characters, unlike the starter's token-aware splitter. "
        "Checklist coverage is a limited diagnostic; inspect the displayed evidence.",
    ]
    for strategy, split in chunks.items():
        target = 500 if strategy == "fixed" else 800
        oversized = sum(len(d.page_content) > target for d in split)
        summary = (
            f"{strategy.title()}: {len(split)} chunks; target={target} characters; "
            f"overlap={0 if strategy == 'fixed' else 150}; "
            f"longest={max(len(d.page_content) for d in split)}; oversized={oversized}."
        )
        print(summary, flush=True)
        settings.append(summary)
    settings.append(
        "CharacterTextSplitter uses its default paragraph separator. A paragraph exceeding "
        "500 characters can remain oversized; 500 is a target, not a guaranteed hard limit."
    )
    if args.check:
        return

    # Load the optional dependency before spending API calls on a full run.
    reranker = load_reranker(args.reranker_model) if args.part != "a" else None
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    stores = {
        strategy: build_or_load_store(split, strategy, embeddings, args.cache_dir)
        for strategy, split in chunks.items()
    }
    numbers = CHUNKING_QUERIES if args.part != "b" else RERANKING_QUERIES
    # Embed each selected question once and reuse it across strategies and bonuses.
    vectors = dict(zip(numbers, embeddings.embed_documents([TEST_QUERIES[n - 1] for n in numbers])))
    if args.part != "b":
        settings.append(compare_chunking(stores, vectors))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n\n".join(settings) + "\n", encoding="utf-8")
        print(f"Bonus A saved to {args.output}", flush=True)
    if args.part != "a":
        settings.append(f"Reranker: `{args.reranker_model}`. LLM: `{args.llm_model}`.")
        settings.append(compare_reranking(
            stores["recursive"], vectors, reranker,
            ChatOpenAI(model=args.llm_model, temperature=0), docs,
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(settings) + "\n", encoding="utf-8")
    print(f"Finished. Full chunks, assessments, and answers: {args.output}", flush=True)


if __name__ == "__main__":
    main()
