"""TF-IDF index over table documents.

Deliberately dependency-free. The corpus here is *ten* table documents, and
pulling scikit-learn (and with it scipy, numpy and pandas, about 340 MB) in to
vectorise ten strings made the deployed image four times larger than the code it
served. This is the same maths, written out.

The weighting matches ``sklearn.feature_extraction.text.TfidfVectorizer`` with
``sublinear_tf=True, smooth_idf=True, norm="l2", ngram_range=(1, 2)``, which is
what this module used before, so linker behaviour is unchanged:

    tf(t, d)  = 1 + ln(count(t, d))
    idf(t)    = ln((1 + N) / (1 + df(t))) + 1
    w(t, d)   = tf * idf, then the document vector is L2-normalised

With both sides L2-normalised, the cosine similarity is just the dot product.
``tests/test_retrieval.py`` pins the weighting, and ``vantage-bench`` reports
linker recall, so a regression here shows up as a red tier rather than a quiet
drop in answer quality.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[a-z0-9]+")

# Light, hand-written stemming. Real analyst questions swing between singular and
# plural constantly ("which category" vs "top categories"), and a full stemmer is
# more machinery than this needs.
_IRREGULAR = {
    "categories": "category",
    "companies": "company",
    "countries": "country",
    "deliveries": "delivery",
    "cities": "city",
}


def normalize(text: str) -> str:
    """Lowercase, split identifiers on underscores, and fold simple plurals."""
    text = text.lower().replace("_", " ")
    out = []
    for tok in _TOKEN.findall(text):
        tok = _IRREGULAR.get(tok, tok)
        if len(tok) > 3 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 3 and tok.endswith("ses"):
            tok = tok[:-2]
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return " ".join(out)


def analyze(text: str, max_ngram: int = 2) -> list[str]:
    """Unigrams and bigrams of the normalised text, in order."""
    tokens = normalize(text).split()
    grams = list(tokens)
    for n in range(2, max_ngram + 1):
        grams += [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return grams


def _l2(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(v * v for v in vector.values()))
    if norm == 0.0:
        return vector
    return {k: v / norm for k, v in vector.items()}


@dataclass
class TfidfIndex:
    """Cosine-similarity search over a fixed set of labelled documents."""

    labels: list[str]
    documents: list[str]
    max_ngram: int = 2
    _idf: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _vectors: list[dict[str, float]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.documents):
            raise ValueError("labels and documents must be the same length")

        counts = [Counter(analyze(doc, self.max_ngram)) for doc in self.documents]
        n_docs = len(counts)

        document_frequency: Counter[str] = Counter()
        for count in counts:
            document_frequency.update(count.keys())

        # smooth_idf: pretend one extra document contains every term, which keeps
        # the denominator non-zero and the weights positive.
        self._idf = {
            term: math.log((1 + n_docs) / (1 + df)) + 1.0
            for term, df in document_frequency.items()
        }
        self._vectors = [
            _l2({term: (1.0 + math.log(n)) * self._idf[term] for term, n in count.items()})
            for count in counts
        ]

    def score(self, query: str) -> dict[str, float]:
        """Cosine similarity of ``query`` against every document, keyed by label."""
        counts = Counter(t for t in analyze(query, self.max_ngram) if t in self._idf)
        if not counts:
            return {label: 0.0 for label in self.labels}
        query_vector = _l2(
            {term: (1.0 + math.log(n)) * self._idf[term] for term, n in counts.items()}
        )
        return {
            label: sum(weight * query_vector.get(term, 0.0) for term, weight in vector.items())
            for label, vector in zip(self.labels, self._vectors)
        }

    def top(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        ranked = sorted(self.score(query).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    @property
    def vocabulary_size(self) -> int:
        return len(self._idf)
