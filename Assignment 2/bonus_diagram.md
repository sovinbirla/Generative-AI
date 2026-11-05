# How `bonus.py` works

This diagram shows the default run, `python bonus.py`, which executes both bonuses.

```mermaid
flowchart TD
    Files["Six airline_kb text files"] --> Load["Load documents and preserve source metadata"]

    subgraph Setup["Shared setup: chunking, caching, and embeddings"]
        Load --> Fixed["Fixed splitter<br/>500 characters, zero overlap"]
        Load --> Recursive["Recursive splitter<br/>800 characters, 150 overlap"]
        Fixed --> Cache["Check separate cache for each strategy<br/>Fingerprint includes chunks, metadata, and embedding model"]
        Recursive --> Cache
        Cache --> Missing{"Any chunks missing?"}
        Missing -->|Yes| Embed["Embed missing chunks<br/>text-embedding-3-small"]
        Embed --> Save["Persist new vectors in Chroma"]
        Missing -->|No| Reuse["Reuse saved document vectors"]
        Save --> Stores[("Separate fixed and recursive Chroma indexes")]
        Reuse --> Stores
        Questions["Selected test questions"] --> QueryVectors["Embed each question once<br/>Reuse its vector across comparisons"]
    end

    subgraph A["Bonus A: compare chunking strategies"]
        RetrieveA["Queries 5, 7, and 10<br/>Retrieve top 3 from EACH index"]
        RetrieveA --> ShowA["Show full chunks, sources, and cosine distances"]
        ShowA --> Evidence["Check expected evidence phrases<br/>Identify facts found and missing"]
        Evidence --> Recommendation["Compare coverage per query<br/>Produce a recommendation"]
    end

    Stores -->|Both indexes| RetrieveA
    QueryVectors --> RetrieveA

    subgraph B["Bonus B: compare answers before and after reranking"]
        Pool["Queries 5 and 10<br/>Retrieve 8 candidates from recursive index"]
        Pool --> Before["BEFORE: keep original top 3"]
        Pool --> CrossEncoder["Cross-encoder scores all 8 query/chunk pairs<br/>ms-marco-MiniLM-L6-v2"]
        CrossEncoder --> After["AFTER: keep reranked top 3"]
        Before --> AnswerBefore["Generate baseline answer with sources<br/>gpt-4o-mini, temperature 0"]
        After --> AnswerAfter["Generate reranked answer with sources<br/>Same model, prompt, and context count"]
        AnswerBefore --> Judge["LLM compares both answers against full reference documents<br/>Discuss correctness, omissions, and citations"]
        AnswerAfter --> Judge
    end

    Stores -->|Recursive index only| Pool
    QueryVectors --> Pool
    Load -.->|Full relevant reference documents| Judge

    Recommendation --> Report["Write bonus_results.md"]
    Judge --> Report
    Pool -.->|Show all 8 original chunks| Report
    After -.->|Show top 3 reranked chunks| Report
```

- **Caching:** saved document vectors are reused; selected questions still need embeddings on each run.
- **Fair answer comparison:** both Bonus B answers receive three chunks selected from the same eight candidates. Reranking changes which chunks are selected and their order.
- **Evaluation:** Bonus A uses a source-specific phrase checklist; Bonus B uses LLM-assisted feedback. These are diagnostics, not guaranteed correctness scores. Lower cosine distance is better; higher cross-encoder score is better.
- **Chunk sizes:** both strategies here measure characters. The fixed splitter's default paragraph separator can produce chunks longer than its 500-character target.
- **Run options:** `--part a` runs only Bonus A; `--part b` runs only Bonus B using the recursive index; `--check` stops after local loading and chunking without API calls.
