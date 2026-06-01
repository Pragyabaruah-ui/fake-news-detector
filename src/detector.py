"""
Fake News Detector - Core ML Pipeline
======================================
Implements a multi-model ensemble for fake news classification
using NLP features: TF-IDF, linguistic patterns, and credibility signals.
"""

import re
import json
import math
import random
from collections import Counter
from typing import Optional


# ---------------------------------------------------------------------------
# Lightweight NLP utilities (no external ML deps required)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","need",
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","his","its","our","their","this","that","these","those",
    "not","no","nor","so","yet","both","either","neither","whether","while",
}

# Patterns associated with sensational / misleading content
SENSATIONAL_PATTERNS = [
    r'\b(BREAKING|URGENT|EXCLUSIVE|SHOCKING|BOMBSHELL|EXPOSED|REVEALED)\b',
    r'\b(you won\'t believe|must see|share before|they don\'t want you to know)\b',
    r'\b(100%|proven|guaranteed|secret|miracle|conspiracy)\b',
    r'[A-Z]{5,}',                         # excessive caps
    r'!{2,}',                              # multiple exclamation marks
    r'\?{2,}',                             # multiple question marks
]

# Credibility signal words (associated with real journalism)
CREDIBILITY_SIGNALS = [
    r'\b(according to|sources say|confirmed|reported|cited|study|research|evidence)\b',
    r'\b(percent|statistics|data|survey|analysis|experts|officials)\b',
    r'\b(published|peer-reviewed|university|institute|department)\b',
]

# Hedge / weasel words (neither good nor bad, but flag them)
HEDGE_WORDS = [
    r'\b(allegedly|reportedly|claimed|unconfirmed|rumored|sources claim)\b',
    r'\b(some say|many believe|people think|could be|might be|possibly)\b',
]


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def count_pattern(text: str, patterns: list[str]) -> int:
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


def sentence_count(text: str) -> int:
    return max(1, len(re.findall(r'[.!?]+', text)))


def avg_word_length(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return sum(len(t) for t in tokens) / len(tokens)


def type_token_ratio(tokens: list[str]) -> float:
    """Lexical diversity: higher = more varied vocabulary."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


# ---------------------------------------------------------------------------
# TF-IDF implementation (no sklearn)
# ---------------------------------------------------------------------------

class TFIDFVectorizer:
    """Minimal TF-IDF with a fixed vocabulary built from training data."""

    def __init__(self, max_features: int = 500):
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def fit(self, corpus: list[str]):
        doc_freq: Counter = Counter()
        tokenized = [tokenize(doc) for doc in corpus]
        n_docs = len(tokenized)

        # document frequency
        for tokens in tokenized:
            for term in set(tokens):
                doc_freq[term] += 1

        # pick top features by df
        top_terms = [t for t, _ in doc_freq.most_common(self.max_features)]
        self.vocab = {t: i for i, t in enumerate(top_terms)}

        # IDF with smoothing
        for term in self.vocab:
            self.idf[term] = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1

    def transform(self, text: str) -> list[float]:
        tokens = tokenize(text)
        tf: Counter = Counter(tokens)
        total = max(1, len(tokens))
        vec = [0.0] * len(self.vocab)
        for term, idx in self.vocab.items():
            if tf[term] > 0:
                vec[idx] = (tf[term] / total) * self.idf.get(term, 1.0)
        return vec


# ---------------------------------------------------------------------------
# Naive Bayes classifier (no sklearn)
# ---------------------------------------------------------------------------

class NaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_log_prior: dict[int, float] = {}
        self.feature_log_prob: dict[int, list[float]] = {}
        self.n_features = 0

    def fit(self, X: list[list[float]], y: list[int]):
        n = len(y)
        classes = list(set(y))
        self.n_features = len(X[0])

        for c in classes:
            indices = [i for i, yi in enumerate(y) if yi == c]
            self.class_log_prior[c] = math.log(len(indices) / n)
            # sum feature values per class
            sums = [0.0] * self.n_features
            for i in indices:
                for j, v in enumerate(X[i]):
                    sums[j] += v
            total = sum(sums) + self.alpha * self.n_features
            self.feature_log_prob[c] = [
                math.log((s + self.alpha) / total) for s in sums
            ]

    def predict_proba(self, x: list[float]) -> dict[int, float]:
        scores = {}
        for c in self.class_log_prior:
            log_prob = self.class_log_prior[c]
            for j, v in enumerate(x):
                if v > 0:
                    log_prob += v * self.feature_log_prob[c][j]
            scores[c] = log_prob

        # softmax
        max_s = max(scores.values())
        exp_s = {c: math.exp(s - max_s) for c, s in scores.items()}
        total = sum(exp_s.values())
        return {c: v / total for c, v in exp_s.items()}

    def predict(self, x: list[float]) -> int:
        proba = self.predict_proba(x)
        return max(proba, key=proba.get)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def extract_linguistic_features(text: str) -> dict:
    tokens = tokenize(text)
    words_raw = re.findall(r'\b\w+\b', text)
    sentences = sentence_count(text)
    words_per_sentence = len(words_raw) / sentences

    sensational_score = count_pattern(text, SENSATIONAL_PATTERNS)
    credibility_score = count_pattern(text, CREDIBILITY_SIGNALS)
    hedge_score = count_pattern(text, HEDGE_WORDS)

    caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    exclamation_ratio = text.count('!') / max(1, len(text))
    question_ratio = text.count('?') / max(1, len(text))

    return {
        "word_count": len(tokens),
        "avg_word_length": avg_word_length(tokens),
        "type_token_ratio": type_token_ratio(tokens),
        "words_per_sentence": words_per_sentence,
        "sensational_score": sensational_score,
        "credibility_score": credibility_score,
        "hedge_score": hedge_score,
        "caps_ratio": caps_ratio,
        "exclamation_ratio": exclamation_ratio,
        "question_ratio": question_ratio,
    }


# ---------------------------------------------------------------------------
# Main Detector class
# ---------------------------------------------------------------------------

class FakeNewsDetector:
    """
    Ensemble classifier combining:
      1. TF-IDF + Naive Bayes (content model)
      2. Linguistic feature heuristics (style model)
    """

    def __init__(self):
        self.vectorizer = TFIDFVectorizer(max_features=300)
        self.nb_model = NaiveBayes(alpha=0.5)
        self.trained = False
        self._train_on_synthetic_data()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_on_synthetic_data(self):
        """
        Bootstrap the model on a curated synthetic dataset that captures
        stylistic differences between real and fake news.
        """
        real_samples = [
            "The Federal Reserve raised interest rates by 25 basis points, according to officials at the central bank. Economists cited inflationary pressures as the primary driver.",
            "Researchers at Stanford University published a peer-reviewed study showing a 15 percent reduction in cardiovascular disease among participants who exercised regularly.",
            "The European Union confirmed new trade regulations affecting imported goods from several Asian markets, with the policy taking effect next quarter.",
            "Scientists confirmed the discovery of a new exoplanet in the habitable zone, according to data published in the journal Nature Astronomy.",
            "The department of transportation reported a 12 percent decline in highway fatalities following the new speed limit enforcement program.",
            "According to the census bureau, the national unemployment rate fell to 3.7 percent last month, beating analyst expectations.",
            "The World Health Organization issued updated guidelines on antibiotic resistance, citing data from 47 member countries.",
            "University researchers confirmed that the new vaccine candidate showed 89 percent efficacy in phase-three clinical trials.",
            "Police officials confirmed the arrest of a suspect in connection with the downtown robbery, citing surveillance evidence.",
            "The finance ministry released quarterly GDP data showing 2.1 percent growth, in line with projections from the IMF.",
            "City engineers completed the infrastructure assessment and identified three bridges requiring urgent maintenance, according to the report.",
            "The supreme court ruled 6-3 on the contested environmental regulation, with justices citing constitutional precedent in their written opinion.",
            "Health authorities reported a measles outbreak affecting 23 individuals in the northern district, and urged vaccination.",
            "The company published its annual sustainability report, confirming a 30 percent reduction in carbon emissions over the past five years.",
            "Olympic committee officials confirmed the host city selection following a closed-door vote among member nations.",
        ]

        fake_samples = [
            "BREAKING: Deep state operatives EXPOSED!! They don't want you to know the TRUTH about what is happening RIGHT NOW!!!",
            "You won't believe what scientists secretly discovered - the miracle cure they've been hiding from you for DECADES!!",
            "URGENT: Share before they delete this!! The government is hiding a SHOCKING conspiracy that affects every single one of us!!!",
            "BOMBSHELL REPORT: Famous celebrity CAUGHT doing something UNBELIEVABLE - the mainstream media refuses to cover this SCANDAL!!!",
            "100% PROVEN: The deep state has been poisoning our water supply!! Whistleblower exposes EVERYTHING - share before it's deleted!!",
            "They're TERRIFIED you'll see this!! The secret vaccine ingredient they don't want you to know about - EXPOSED!!",
            "Wake up sheeple!! The moon landing was TOTALLY FAKED and here's the UNDENIABLE PROOF they've been hiding since 1969!!!",
            "MUST SHARE IMMEDIATELY: Scientists discover aliens are ALREADY LIVING AMONG US and the government knows everything!!",
            "SHOCKING: Famous billionaire secretly controls ALL media and this leaked document PROVES IT beyond any doubt!!!",
            "The cure for cancer has existed since 1930 but big pharma is SUPPRESSING it to keep making billions off your suffering!!!",
            "EXPOSED: Every single election for the past 50 years has been RIGGED!! This whistleblower has the EVIDENCE they fear!!",
            "You're being LIED TO about climate change - the REAL agenda is about TOTAL CONTROL of every human being on earth!!!",
            "BREAKING BOMBSHELL: The president's secret twin has been running the country for years - PROOF INSIDE!!!",
            "They banned this video EVERYWHERE but we have it!! The technology that makes oil companies obsolete - SHARE NOW!!!",
            "This natural herb CURES diabetes in 72 hours but doctors REFUSE to tell you because it would destroy their industry!!!",
        ]

        corpus = real_samples + fake_samples
        labels = [0] * len(real_samples) + [1] * len(fake_samples)  # 0=real, 1=fake

        # Fit TF-IDF
        self.vectorizer.fit(corpus)

        # Build feature matrix (TF-IDF + linguistic)
        X = []
        for text in corpus:
            tfidf_vec = self.vectorizer.transform(text)
            ling = extract_linguistic_features(text)
            ling_vec = list(ling.values())
            X.append(tfidf_vec + ling_vec)

        self.nb_model.fit(X, labels)
        self.trained = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _linguistic_heuristic_score(self, features: dict) -> float:
        """
        Rule-based score in [0, 1] where 1 = very likely fake.
        """
        score = 0.0

        # Sensationalism strongly associated with fake
        score += min(features["sensational_score"] * 0.15, 0.45)

        # Credibility signals lower the fake score
        score -= min(features["credibility_score"] * 0.08, 0.24)

        # High caps ratio
        score += min(features["caps_ratio"] * 2.0, 0.20)

        # Exclamation marks
        score += min(features["exclamation_ratio"] * 15, 0.15)

        # Shorter, simpler words can indicate tabloid style
        if features["avg_word_length"] < 4.5:
            score += 0.05

        # Very low lexical diversity
        if features["type_token_ratio"] < 0.35:
            score += 0.05

        return max(0.0, min(1.0, score))

    def analyze(self, title: str, body: str = "") -> dict:
        """
        Full analysis of an article. Returns structured results.
        """
        full_text = f"{title} {body}".strip()
        if not full_text:
            return {"error": "No text provided"}

        # TF-IDF + NB prediction
        tfidf_vec = self.vectorizer.transform(full_text)
        ling_features = extract_linguistic_features(full_text)
        ling_vec = list(ling_features.values())
        feature_vec = tfidf_vec + ling_vec

        nb_proba = self.nb_model.predict_proba(feature_vec)
        nb_fake_prob = nb_proba.get(1, 0.5)

        # Heuristic score
        heuristic_fake_prob = self._linguistic_heuristic_score(ling_features)

        # Ensemble (weighted average)
        ensemble_fake_prob = 0.55 * nb_fake_prob + 0.45 * heuristic_fake_prob

        # Classification
        if ensemble_fake_prob >= 0.65:
            verdict = "FAKE"
            confidence = "High" if ensemble_fake_prob >= 0.80 else "Medium"
        elif ensemble_fake_prob >= 0.45:
            verdict = "SUSPICIOUS"
            confidence = "Low"
        else:
            verdict = "LIKELY REAL"
            confidence = "High" if ensemble_fake_prob <= 0.25 else "Medium"

        # Identify specific red flags
        red_flags = []
        if ling_features["sensational_score"] > 0:
            red_flags.append(f"Sensational language detected ({ling_features['sensational_score']} instances)")
        if ling_features["caps_ratio"] > 0.05:
            red_flags.append(f"Excessive capitalization ({ling_features['caps_ratio']:.1%} of text)")
        if ling_features["exclamation_ratio"] > 0.005:
            red_flags.append("Heavy use of exclamation marks")
        if ling_features["credibility_score"] == 0 and len(full_text) > 200:
            red_flags.append("No credibility signals (citations, sources, data)")
        if ling_features["type_token_ratio"] < 0.35:
            red_flags.append("Low lexical diversity (repetitive language)")

        # Positive signals
        green_flags = []
        if ling_features["credibility_score"] > 0:
            green_flags.append(f"Contains credibility signals ({ling_features['credibility_score']} found)")
        if ling_features["avg_word_length"] >= 5.0:
            green_flags.append("Sophisticated vocabulary")
        if ling_features["words_per_sentence"] >= 15:
            green_flags.append("Complex sentence structure")
        if ling_features["hedge_score"] > 0:
            green_flags.append("Uses appropriate hedging language")

        return {
            "verdict": verdict,
            "confidence": confidence,
            "fake_probability": round(ensemble_fake_prob * 100, 1),
            "real_probability": round((1 - ensemble_fake_prob) * 100, 1),
            "model_scores": {
                "naive_bayes": round(nb_fake_prob * 100, 1),
                "linguistic_heuristic": round(heuristic_fake_prob * 100, 1),
                "ensemble": round(ensemble_fake_prob * 100, 1),
            },
            "linguistic_features": {k: round(v, 4) if isinstance(v, float) else v
                                     for k, v in ling_features.items()},
            "red_flags": red_flags,
            "green_flags": green_flags,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detector = FakeNewsDetector()

    test_cases = [
        {
            "title": "Federal Reserve raises interest rates amid inflation concerns",
            "body": "The Federal Reserve raised interest rates by 25 basis points on Wednesday, according to officials. Economists cited persistent inflationary pressures as the primary driver of the decision.",
        },
        {
            "title": "BREAKING: Deep State EXPOSED!! Share Before They Delete This!!",
            "body": "You WON'T BELIEVE what they've been hiding from you!!! 100% PROVEN conspiracy that the mainstream media REFUSES to cover!! WAKE UP SHEEPLE!!!",
        },
        {
            "title": "Scientists reportedly discover new treatment, but details remain unconfirmed",
            "body": "Some researchers claim to have found a potential treatment, though the findings have not yet been peer-reviewed. Experts urge caution pending further analysis.",
        },
    ]

    print("\n" + "="*60)
    print("  FAKE NEWS DETECTOR — Analysis Results")
    print("="*60)

    for case in test_cases:
        result = detector.analyze(case["title"], case["body"])
        print(f"\n📰 Title: {case['title'][:70]}...")
        print(f"   Verdict:     {result['verdict']} ({result['confidence']} confidence)")
        print(f"   Fake Score:  {result['fake_probability']}%")
        print(f"   Red Flags:   {len(result['red_flags'])}")
        if result["red_flags"]:
            for flag in result["red_flags"]:
                print(f"     ⚠  {flag}")
        if result["green_flags"]:
            for flag in result["green_flags"]:
                print(f"     ✓  {flag}")

    print("\n" + "="*60 + "\n")
