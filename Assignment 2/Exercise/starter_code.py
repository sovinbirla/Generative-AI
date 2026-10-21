"""
RAG Pipeline Starter Code — Airline Operations Assistant
ADSP 32027: Generative AI Principles

Instructions:
- Fill in every section marked with TODO
- Run checkpoints after each stage to verify your work
- Do not modify the checkpoint print statements — they are used for grading
"""

from pathlib import Path
from dotenv import load_dotenv

# ── Install required packages (run once in terminal) ──────────────────────────
# pip install langchain langchain-openai langchain-community chromadb tiktoken python-dotenv

# ── Load your local API key (the .env file is ignored by Git) ─────────────────
load_dotenv(Path(__file__).with_name(".env"))

# =============================================================================
# STAGE 1: DOCUMENT LOADING
# =============================================================================
# Goal: Load all .txt files from the airline_kb/ folder.
# Each document should carry its filename as metadata (source field).

def load_documents(kb_dir: str = "airline_kb"):
    """
    Load all .txt files from kb_dir.

    Hint: Use DirectoryLoader from langchain_community.document_loaders
    with glob="*.txt" and loader_cls=TextLoader.

    Returns:
        List of LangChain Document objects
    """
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    loader = DirectoryLoader(
        kb_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    # ── Checkpoint 1 ──────────────────────────────────────────────────────────
    print(f"\n[Checkpoint 1] Loaded {len(docs)} documents")
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"[Checkpoint 1] Total characters: {total_chars}")
    for doc in docs:
        print(f"  - {doc.metadata.get('source', 'unknown')} ({len(doc.page_content)} chars)")

    return docs


# =============================================================================
# STAGE 2: CHUNKING
# =============================================================================
# Goal: Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

def chunk_documents(docs, chunk_size: int = 800, chunk_overlap: int = 150):
    """
    Split documents into overlapping chunks measured in tokens.

    Hint: Use RecursiveCharacterTextSplitter. Make sure source metadata
    is preserved in each chunk (LangChain does this automatically).

    Returns:
        List of LangChain Document chunks
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            r"\n\s*\n",        # Paragraph boundaries
            r"(?<=[.!?])\s+",  # Whitespace after sentence-ending punctuation
            r"\n",            # Line boundaries
            r"\s+",           # Word boundaries
            "",               # Character fallback for oversized text
        ],
        is_separator_regex=True,
        keep_separator="end",
    )

    chunks = splitter.split_documents(docs)

    # ── Checkpoint 2 ──────────────────────────────────────────────────────────
    print(f"\n[Checkpoint 2] Created {len(chunks)} chunks")
    print(f"[Checkpoint 2] Sample chunk (first 200 chars):\n  {chunks[0].page_content[:200]!r}")
    print(f"[Checkpoint 2] Chunk source: {chunks[0].metadata.get('source', 'unknown')}")

    return chunks


# =============================================================================
# STAGE 3 + 4: EMBEDDINGS AND VECTOR STORE
# =============================================================================
# Goal: Embed the chunks and store them in a persistent ChromaDB vector store.

CHROMA_DIR = "chroma_db"

def build_vector_store(chunks, persist_dir: str = CHROMA_DIR):
    """
    Create embeddings and build a Chroma vector store.

    Option A (requires OpenAI key):
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    Option B (free, no API key needed):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    Hint: Use Chroma.from_documents(..., persist_directory=persist_dir)

    Returns:
        Chroma vector store object
    """
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    # TODO: Choose and instantiate your embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # replace with your embedding model

    # TODO: Create the Chroma vector store from chunks
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)  # replace with Chroma.from_documents(...)

    # ── Checkpoint 3 ──────────────────────────────────────────────────────────
    count = vectorstore._collection.count()
    print(f"\n[Checkpoint 3] Vector store contains {count} vectors")
    sample_emb = embeddings.embed_query("test")
    print(f"[Checkpoint 3] Embedding dimension: {len(sample_emb)}")

    return vectorstore


def load_vector_store(persist_dir: str = CHROMA_DIR):
    """Load an existing Chroma vector store (avoids re-embedding on repeat runs)."""
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    # TODO: Choose the SAME embedding model you used in build_vector_store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # replace with your embedding model

    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


# =============================================================================
# STAGE 5: RAG CHAIN
# =============================================================================
# Goal: Build a retrieval chain that fetches relevant chunks and generates
# a grounded, source-cited answer.

def build_rag_chain(vectorstore, top_k: int = 4):
    """
    Build a RetrievalQAWithSourcesChain.

    Requirements:
    1. Retriever: similarity search, k=top_k
    2. LLM: ChatOpenAI, temperature=0 (for reproducibility)
    3. Custom prompt that:
       a. Instructs the model to answer from provided context only
       b. Requires a source document citation
       c. Handles the "not in context" case

    Hint: Use PromptTemplate with input_variables=["summaries", "question"]
    and pass it via chain_type_kwargs={"prompt": prompt}

    Returns:
        RetrievalQAWithSourcesChain
    """
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQAWithSourcesChain
    from langchain.prompts import PromptTemplate

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt_template = """
    You are an expert airline operations assistant.
    Use only the context below to answer the question.
    At the end of your answer, cite the source document(s) on a separate
    line beginning with "SOURCES:".
    If the answer is not in the context, say:
    "This information is not available in the operational documents."

    Context:
    {summaries}

    Question: {question}

    Answer (with source citation):"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["summaries", "question"]
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )

    # ── Checkpoint 4: Retrieval Test ──────────────────────────────────────────
    test_query = "What happens when a flight is mechanically delayed?"
    retrieved = retriever.invoke(test_query)
    print(f"\n[Checkpoint 4] Retrieval test for: '{test_query}'")
    for i, doc in enumerate(retrieved, 1):
        src = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i}] Source: {src} | Preview: {preview!r}")

    return chain


# =============================================================================
# QUERY INTERFACE
# =============================================================================

def answer_query(chain, question: str) -> dict:
    """Run a single query through the RAG chain and return answer + sources."""
    result = chain.invoke({"question": question})
    return {
        "question": question,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", ""),
    }


# =============================================================================
# MAIN
# =============================================================================

TEST_QUERIES = [
    "What are the immediate steps an agent should take when a flight is delayed due to a mechanical issue?",
    "Under what weather conditions does the airline automatically rebook passengers on the next available flight?",
    "What is the maximum crew duty period allowed before a mandatory rest period is required?",
    "Explain the priority order for gate reassignment during irregular operations.",
    "How should ground staff communicate a delay of more than 2 hours to connecting passengers?",
    "What documentation is required when a flight is cancelled due to Air Traffic Control (ATC) restrictions?",
    "Compare the rebooking options available to elite status passengers versus standard passengers during a weather cancellation.",
    "What are the de-icing requirements for departures in freezing rain conditions?",
    "Describe the escalation process when a mechanical delay extends beyond the originally estimated time.",
    "How does crew rest regulation compliance affect flight scheduling recovery during a major hub disruption?",
]


if __name__ == "__main__":
    KB_DIR = "airline_kb"

    # Build or load the vector store
    if Path(CHROMA_DIR).exists():
        print("Loading existing vector store...")
        vectorstore = load_vector_store()
    else:
        docs = load_documents(KB_DIR)
        chunks = chunk_documents(docs)
        vectorstore = build_vector_store(chunks)

    # Build the RAG chain (Checkpoint 4 runs here)
    chain = build_rag_chain(vectorstore)

    # ── Checkpoint 5: Full pipeline test ─────────────────────────────────────
    print("\n[Checkpoint 5] Running full pipeline on sample query...")
    sample = answer_query(chain, TEST_QUERIES[0])
    print(f"  Q: {sample['question']}")
    print(f"  A: {sample['answer']}")
    print(f"  Sources: {sample['sources']}")

    # Run all test queries
    print("\n" + "=" * 70)
    print("RUNNING ALL TEST QUERIES")
    print("=" * 70)
    for query in TEST_QUERIES:
        result = answer_query(chain, query)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 70)
