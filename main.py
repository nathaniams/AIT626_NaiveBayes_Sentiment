"""Student 3: integrate teammates' preprocessing/model, evaluate 4 required NB variants.

Run from repository root:
    python main.py --data-dir data --output-dir outputs
Optional bonus:
    python main.py --data-dir data --output-dir outputs --bonus

Expects existing src/preprocessing.py and src/model.py from teammates.
"""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.preprocessing import (
    build_vocabulary_and_features, preprocess_text, create_count_vector,
)
from src.model import NaiveBayesClassifier
from src.evaluation import format_report


LABELS = (('negative', 0), ('positive', 1))


def load_test_documents(test_dir, do_stem):
    """Load held-out examples WITHOUT building vocabulary from the test set."""
    documents, labels = [], []
    for label_name, label in LABELS:
        folder = test_dir / label_name
        paths = sorted(folder.glob('*.txt'))
        if not paths:
            raise FileNotFoundError(f'No .txt files found under {folder}')
        for path in paths:
            content = path.read_text(encoding='utf-8', errors='ignore')
            documents.append(preprocess_text(content, do_stem=do_stem))
            labels.append(label)
    return documents, labels


def make_vectors(documents, vocab_set, mode, idf=None):
    """Sparse vectors using Student 1's count extractor.

    Binary stores ONLY present words with weight 1, instead of creating
    a massive dense dictionary of all vocabulary words for every tweet.
    """
    vectors = []
    for tokens in documents:
        counts = dict(create_count_vector(tokens, vocab_set))
        if mode == 'count':
            vec = counts
        elif mode == 'binary':
            vec = {word: 1 for word in counts}
        elif mode == 'tfidf':
            if idf is None:
                raise ValueError('TF-IDF requires training-derived IDF values')
            vec = {word: count * idf[word] for word, count in counts.items()}
        else:
            raise ValueError(f'Unknown mode: {mode}')
        vectors.append(vec)
    return vectors


def training_idf(training_documents, vocab_set):
    """Smooth IDF computed exclusively on training data (optional bonus)."""
    import math
    df = Counter()
    for doc in training_documents:
        df.update(set(doc) & vocab_set)
    n_docs = len(training_documents)
    return {word: math.log((n_docs + 1) / (df[word] + 1)) + 1.0
            for word in vocab_set}


def run_experiments(data_dir, output_dir, bonus=False):
    train_dir, test_dir = data_dir / 'train', data_dir / 'test'
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f'Expected data/train and data/test under {data_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    summaries = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summaries.append('AIT 626 Naive Bayes Sentiment Classification')
    summaries.append(f'Run timestamp: {timestamp}')
    summaries.append('Labels: 0=negative, 1=positive')
    summaries.append('All vocabularies and optional IDF derived from training set only.')
    summaries.append('Multinomial NB with Laplace alpha=1; binary means binary-weighted Multinomial NB.')
    summaries.append('')

    # Two stemming settings x two feature settings = 4 mandatory experiments.
    for do_stem in (False, True):
        stem_label = 'stem' if do_stem else 'no_stem'
        vocab, train_docs, train_labels = build_vocabulary_and_features(
            str(train_dir), do_stem=do_stem
        )
        vocab_set = set(vocab)
        if not train_docs or not vocab_set:
            raise ValueError(f'Empty train documents or vocabulary for {stem_label}')
        if set(train_labels) != {0, 1}:
            raise ValueError('Training set must include positive and negative reviews')
        test_docs, test_labels = load_test_documents(test_dir, do_stem=do_stem)
        print(f'[{stem_label}] train={len(train_docs)}, test={len(test_docs)}, |V|={len(vocab_set)}')
        summaries.append(
            f'{stem_label}: training documents={len(train_docs)}, '
            f'test documents={len(test_docs)}, |V|={len(vocab_set)}'
        )

        modes = ['count', 'binary'] + (['tfidf'] if bonus else [])
        idf = training_idf(train_docs, vocab_set) if bonus else None
        for mode in modes:
            experiment = f'{stem_label}_{mode}'
            x_train = make_vectors(train_docs, vocab_set, mode, idf=idf)
            x_test = make_vectors(test_docs, vocab_set, mode, idf=idf)
            model = NaiveBayesClassifier(alpha=1.0)
            model.fit(x_train, train_labels, vocab_set)
            predictions = model.predict(x_test)
            report, accuracy, cm = format_report(
                experiment, test_labels, predictions, model.summary()
            )
            summaries.extend([report, '-' * 64, ''])
            results.append((experiment, accuracy))
            print(f'{experiment}: accuracy={accuracy:.4f}, cm={cm}')

    summaries.extend(['SUMMARY', 'Experiment                      Accuracy', '-' * 42])
    for experiment, accuracy in results:
        summaries.append(f'{experiment:<30} {accuracy:.4f}')
    if bonus:
        summaries.append('')
        summaries.append('Bonus: raw term frequency * smoothed training-set IDF;')
        summaries.append('fractional TF-IDF weights in Multinomial NB are an exploratory extension.')
    log_path = output_dir / 'results.txt'
    log_path.write_text('\n'.join(summaries) + '\n', encoding='utf-8')
    print(f'\nSaved all experiment results to: {log_path.resolve()}')
    return results, log_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate four Naive Bayes sentiment configurations')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--bonus', action='store_true', help='Also evaluate TF-IDF weighted NB')
    args = parser.parse_args()
    run_experiments(args.data_dir, args.output_dir, args.bonus)


if __name__ == '__main__':
    main()
