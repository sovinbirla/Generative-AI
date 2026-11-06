"""Build or reuse the local semantic index. Network downloads require an explicit flag."""
import argparse
from .retrieval import Retriever, load_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-model', action='store_true')
    args = parser.parse_args()
    retriever = Retriever(load_corpus(), allow_download=args.download_model)
    if retriever.semantic_error:
        raise SystemExit('Semantic index was not created: ' + retriever.semantic_error)
    print(f'Indexed {len(retriever.chunks)} chunks. {retriever.mode}; fingerprint={retriever.fingerprint}')


if __name__ == '__main__':
    main()
