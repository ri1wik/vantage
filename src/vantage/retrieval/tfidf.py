"""TF-IDF index over table documents.

Small, deterministic and dependency-light on purpose: the schema linker has to
produce the same table set on every bench run, so nothing here touches a network
or a random seed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer

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


@dataclass
class TfidfIndex:
    """Cosine-similarity search over a fixed set of labelled documents."""

    labels: list[str]
    documents: list[str]

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.documents):
            raise ValueError("labels and documents must be the same length")
        self._vectorizer = TfidfVectorizer(
            preprocessor=normalize,
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(self.documents)

    def score(self, query: str) -> dict[str, float]:
        """Cosine similarity of ``query`` against every document, keyed by label."""
        if not query.strip():
            return {label: 0.0 for label in self.labels}
        vec = self._vectorizer.transform([query])
        sims = (self._matrix @ vec.T).toarray().ravel()
        return {label: float(s) for label, s in zip(self.labels, sims)}

    def top(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        ranked = sorted(self.score(query).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    @property
    def vocabulary_size(self) -> int:
        return len(self._vectorizer.vocabulary_)
