import os
import glob
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict

nltk.download('punkt')
nltk.download('punkt_tab')

def preprocess_text(text, do_stem = False):
    """
    1. Remove HTML markup tags using BeautifulSoup.
    2. Tokenizes text at whitespace and punctuations.
    3. Lowercase word starting with a capital, but keep all capatilized words as is (e.g., USA).
    4. Optionally applies Porter stemming.
    """
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text()

    tokens = word_tokenize(clean_text)
    processed_tokens = []
    stemmer = PorterStemmer() if do_stem else None

    for token in tokens:
        # Lowercase captialized words, but keep all capilized words as is
        if token.istitle() and not token.isupper():
            token = token.lower()

        if do_stem:
            token = stemmer.stem(token)

        processed_tokens.append(token)

    return processed_tokens

def build_vocabulary_and_features(data_dir, do_stem = False):
    """
    Reads training data word by word and creates vocabulary (V).
    Use 'postive' and 'negative' subfolders.
    """
    vocab = set()
    documents = []
    labels = []

    # Map classes to indices (Positive = 1, Negative = 0)
    class_mapping = {'positive': 1, 'negative': 0}

    for folder_name, label_idx in class_mapping.items():
        class_path = os.path.join(data_dir, folder_name, '*.txt')
        for filepath in glob.glob(class_path):
            with open(filepath, 'r', encoding = 'utf-8', errors = 'ignore') as f:
                content = f.read()
                tokens = preprocess_text(content, do_stem = do_stem)
                vocab.update(tokens)
                documents.append(tokens)
                labels.append(label_idx)

    return list(vocab), documents, labels

def create_count_vector(tokens, vocab):
    """Creates a Frequency Count vector representing word counts per document."""
    vector = defaultdict(int)
    for token in tokens:
        if token in vocab:
            vector[token] += 1
    return vector

def create_binary_vector(tokens, vocab):
    """Creates a binary vector tracking presence (1) or absense (0) of each word."""
    vector = {}
    token_set = set(tokens)
    for word in vocab:
        vector[word] = 1 if word in token_set else 0
    return vector

## Test preprocessing
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(current_dir, '..', 'data', 'train')

    print(f"Testing data loader with path: {train_dir}")

    # Test building vocab and loading documents without stemming
    vocab, documents, labels, = build_vocabulary_and_features(train_dir, do_stem = True)

    print(f"Successfully loaded {len(documents)} training documents.")
    print(f"Vocabulary size (No Stemming): {len(vocab)}")
    print(f"First 5 labels: {labels[:5]}")

    # Test feature extraction on the first document
    if documents:
        sample_tokens = documents[0]
        count_vec = create_count_vector(sample_tokens, vocab)
        binary_vec = create_binary_vector(sample_tokens, vocab)

        print(f"Sample document token count: {len(sample_tokens)}")
        print(f"Count vector non-zero entries: {len(count_vec)}")
        print(f"Binary vector total entries: {len(binary_vec)}")
    print("Preprocessing test completed successfully.")