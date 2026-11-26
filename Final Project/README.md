# Multimodal Conversational AI for E-commerce

The ICML-style paper, **Multimodal Catalog Retrieval and Grounded Product QA.pdf**, contains the system design, preprocessing decisions, evaluation protocol, measured results, error analysis, and reproduction appendix.

## Run the chatbot

From this folder on the current machine, use the existing course environment.

    ../.venv/bin/python -m streamlit run app.py

On another machine with Python 3.12, create an environment with the pinned packages.

    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m streamlit run app.py

Open the local URL printed by Streamlit. The first turn loads the models. No API key is needed when the local checkpoints and index are present.

1. Ask “Show me a picture of Oasis Supply Spiral Birthday Candles, 2.25-Inch” to retrieve the listing and open its image card.
2. Enable **Use previous product matches for this follow-up** and ask “What is the listed price?” The source lists $71.42 in the January 2020 snapshot.
3. Upload **data/images/B0044Z90LM-0.jpg** without text to try image-based identification, or add a question about the uploaded product. Each answer exposes the retrieved listing cards.
4. Choose a prompt variant in the sidebar and use **Clear conversation** to start over.

The catalog is a historical sample. The source cards let you inspect the listing matched to a photo.

## Rebuild the index, evaluation and paper

The local copy includes the source CSV, cached photos, normalized records, SQLite index and checkpoint files. If copying only the code, obtain the January 2020 CSV from the [assigned Kaggle dataset](https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020) and save it as **data/amazon.csv**. The environment variables **CLIP_MODEL** and **QA_MODEL** can point to local checkpoints or to **openai/clip-vit-base-patch32** and **Qwen/Qwen2.5-0.5B-Instruct** for an initial online download.

From this folder with your Python environment active, run the following.

    python catalog.py --limit 240 --seed 42 --download-images
    python rag.py index
    python evaluate.py
    python paper/render_results.py
    cd paper
    latexmk -pdf main.tex

The first command skips cached photos. The external photo URLs may change or fail in later reproductions. Evaluation takes several minutes on CPU. **results/run.json** records the benchmark data fingerprints. The report uses the official ICML 2025 LaTeX template with a course-project notice. [ICML author instructions](https://icml.cc/Conferences/2025/AuthorInstructions)

- **results/retrieval_metrics.csv** contains Recall@1, Recall@5, Recall@10, MRR@10, and query latency.
- **results/qa_metrics.csv** contains the relevance and correctness proxies, target-in-context rate, grounded price rate, and generation latency.
- **results/retrieval_cases.csv** and **results/qa_cases.csv** contain the rankings and generated responses for audit.
