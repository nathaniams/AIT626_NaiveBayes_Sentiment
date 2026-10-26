
import math
from collections import defaultdict


class NaiveBayesClassifier:
    """Multinomial Naive Bayes with add-alpha smoothing, computed in log space."""

    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("alpha must be > 0, otherwise unseen words get zero probability")
        self.alpha = alpha
        self.classes = []                        
        self.vocab = set()
        self.log_prior = {}                      
        self.log_likelihood = {}                 
        self.log_unseen = {}                     
        self.class_doc_count = {}                
        self.class_total_weight = {}             

    
    # Training
    
    def fit(self, vectors, labels, vocab):
        """
        Train the classifier.

        vectors : list of dicts, one per training document, {word: weight}.
                  Output of create_count_vector / create_binary_vector
                  (zero entries are skipped automatically).
        labels  : list of class labels (1 = positive, 0 = negative).
        vocab   : iterable of training-vocabulary words (V).
        """
        if len(vectors) != len(labels):
            raise ValueError("vectors and labels must have the same length")
        if not vectors:
            raise ValueError("cannot train on an empty dataset")

        self.vocab = set(vocab)
        V = len(self.vocab)
        n_docs = len(labels)
        self.classes = sorted(set(labels))

        # Sum the feature weights of each word, separately for each class
        word_weight = {c: defaultdict(float) for c in self.classes}
        doc_count = {c: 0 for c in self.classes}

        for vec, label in zip(vectors, labels):
            doc_count[label] += 1
            for word, weight in vec.items():
                if weight and word in self.vocab:
                    word_weight[label][word] += weight

        for c in self.classes:
            # Prior: fraction of training documents in class c
            self.class_doc_count[c] = doc_count[c]
            self.log_prior[c] = math.log(doc_count[c] / n_docs)

            # Likelihood denominator: total weight in class c + alpha * |V|
            total = sum(word_weight[c].values())
            self.class_total_weight[c] = total
            denom = total + self.alpha * V

            # Store log P(w|c) only for words seen in c; every other word in V
            # shares the same smoothed value alpha / denom
            self.log_likelihood[c] = {
                w: math.log((wt + self.alpha) / denom)
                for w, wt in word_weight[c].items()
            }
            self.log_unseen[c] = math.log(self.alpha / denom)

        return self

    def fit_tokens(self, documents, labels, vocab, binary=False):
        """
        Convenience wrapper: train directly from token lists.

        binary=False -> frequency counts; binary=True -> presence (each word
        counted at most once per document, i.e. Binary Multinomial NB).
        """
        return self.fit([self.tokens_to_vector(t, binary) for t in documents],
                        labels, vocab)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def word_log_likelihood(self, word, c):
        """log P(word | c) with smoothing; assumes word is in the vocabulary."""
        return self.log_likelihood[c].get(word, self.log_unseen[c])

    def class_scores(self, vector):
        """
        Return {class: log P(c) + sum weight * log P(w|c)} for one document.
        Words outside the training vocabulary are ignored.
        """
        self._check_fitted()
        scores = {}
        for c in self.classes:
            score = self.log_prior[c]
            for word, weight in vector.items():
                if weight and word in self.vocab:
                    score += weight * self.word_log_likelihood(word, c)
            scores[c] = score
        return scores

    def predict_one(self, vector):
        """Most likely class for one document vector (ties go to the larger label)."""
        scores = self.class_scores(vector)
        return max(self.classes, key=lambda c: (scores[c], c))

    def predict(self, vectors):
        """Most likely class for each document vector."""
        return [self.predict_one(v) for v in vectors]

    def predict_tokens(self, documents, binary=False):
        """Predict directly from token lists (use the same binary flag as training)."""
        return self.predict([self.tokens_to_vector(t, binary) for t in documents])

    
    # Helpers for reporting / documentation
    
    @staticmethod
    def tokens_to_vector(tokens, binary=False):
        """Token list -> sparse {word: weight} dict (counts or binary)."""
        vec = defaultdict(int)
        for t in tokens:
            vec[t] += 1
        if binary:
            return {w: 1 for w in vec}
        return dict(vec)

    def most_informative_words(self, n=10):
        """
        Words with the largest log-likelihood ratio between the two classes.
        Returns {class: [(word, log P(w|c) - log P(w|other)), ...]}.
        Useful for the README / output log to show what the model learned.
        """
        self._check_fitted()
        if len(self.classes) != 2:
            raise ValueError("most_informative_words is defined for two classes")
        a, b = self.classes
        ratios = {w: self.word_log_likelihood(w, a) - self.word_log_likelihood(w, b)
                  for w in self.vocab}
        ranked = sorted(ratios.items(), key=lambda kv: kv[1])
        return {a: [(w, r) for w, r in reversed(ranked[-n:])],
                b: [(w, -r) for w, r in ranked[:n]]}

    def summary(self):
        """Short text summary of the trained model for the output log."""
        self._check_fitted()
        lines = [f"Naive Bayes (alpha = {self.alpha}), |V| = {len(self.vocab)}"]
        n_docs = sum(self.class_doc_count.values())
        for c in self.classes:
            lines.append(
                f"  class {c}: {self.class_doc_count[c]} docs, "
                f"prior = {self.class_doc_count[c]}/{n_docs} = {math.exp(self.log_prior[c]):.4f}, "
                f"total feature weight = {self.class_total_weight[c]:g}"
            )
        return "\n".join(lines)

    def _check_fitted(self):
        if not self.classes:
            raise RuntimeError("model is not trained yet; call fit() first")



# Self-test with the worked example from Jurafsky & Martin, SLP3 Ch. 4

if __name__ == "__main__":
    train = [
        ("just plain boring", 0),
        ("entirely predictable and lacks energy", 0),
        ("no surprises and very few laughs", 0),
        ("very powerful", 1),
        ("the most fun film of the summer", 1),
    ]
    docs = [s.split() for s, _ in train]
    labels = [y for _, y in train]
    vocab = {w for d in docs for w in d}

    nb = NaiveBayesClassifier(alpha=1.0).fit_tokens(docs, labels, vocab)
    print(nb.summary())

    test = "predictable with no fun".split()          
    scores = nb.class_scores(nb.tokens_to_vector(test))
    print(f"P(-)P(S|-) = {math.exp(scores[0]):.2e}   (textbook: 6.1e-05)")
    print(f"P(+)P(S|+) = {math.exp(scores[1]):.2e}   (textbook: 3.2e-05)")
    print("Predicted:", "positive" if nb.predict_one(nb.tokens_to_vector(test)) else "negative")
