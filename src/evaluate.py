"""
Evaluation Module — Fake News Detector
========================================
Provides cross-validation, metrics, and a benchmark report.
"""

import math
import random
from src.detector import FakeNewsDetector

# ------------------------------------------------------------------
# Benchmark dataset (labeled examples)
# ------------------------------------------------------------------

BENCHMARK_DATA = [
    # (title, body, label)  label: 0=real, 1=fake
    ("Scientists publish new climate study in Nature journal",
     "Researchers at MIT published peer-reviewed findings showing a 1.2°C rise in average temperatures over the last decade, citing satellite data and weather station records.",
     0),
    ("Central bank confirms interest rate decision following board meeting",
     "Officials confirmed the 50-basis-point rate cut following a two-day policy meeting. Economists and analysts had widely expected the move amid slowing growth data.",
     0),
    ("University study links exercise to improved cognitive function",
     "A randomized controlled trial of 1,200 participants found that 30 minutes of daily aerobic exercise improved memory test scores by 18 percent, according to published results.",
     0),
    ("City council votes to expand public transport network",
     "The council voted 8-3 in favor of the expansion project, which will add 12 new bus routes and extend the metro line by six kilometers.",
     0),
    ("Health ministry reports decline in infectious disease cases",
     "The ministry of health reported a 22 percent drop in reported flu cases compared to the same period last year, citing higher vaccination uptake.",
     0),
    ("Trade agreement signed between two nations after years of negotiations",
     "Officials from both governments signed the comprehensive trade deal on Thursday, with the agreement expected to reduce tariffs on over 400 goods.",
     0),
    ("BOMBSHELL: Famous Politician SECRETLY Controls Global Banks — EXPOSED!!!",
     "The truth they NEVER wanted you to know!! Share before it gets deleted!! 100% PROVEN with undeniable evidence that will SHOCK you!!!",
     1),
    ("You WON'T BELIEVE what the government is putting in your water!!",
     "WAKE UP SHEEPLE!! They've been poisoning us for DECADES and the mainstream media is COMPLETELY SILENT about it!! SHARE THIS NOW before it disappears!!",
     1),
    ("MIRACLE CURE Doctors DON'T Want You to Know About — It WORKS!!",
     "Big Pharma has been HIDING this natural remedy for 50 YEARS because it would DESTROY their billion-dollar industry!! 100% PROVEN to cure everything!!!",
     1),
    ("BREAKING: Moon Landing Was TOTALLY FAKED — Here's the PROOF!!",
     "Exclusive footage EXPOSES the greatest lie in human history!! The deep state has been covering this up for decades but we have the UNDENIABLE EVIDENCE!!!",
     1),
    ("SHOCKING Secret Exposed: What They're REALLY Putting in Vaccines!!!",
     "You won't believe the TRUTH they've been hiding!! Share immediately before they ban this!! Whistleblower reveals EVERYTHING the government fears!!",
     1),
    ("Billionaire SECRETLY runs shadow government — leaked documents PROVE it!!",
     "BOMBSHELL REVELATION: A secret cabal of billionaires has been controlling every election for 30 years!! The evidence is UNDENIABLE and they're TERRIFIED!!!",
     1),
    ("Researchers allegedly find new exoplanet, though confirmation pending",
     "Scientists claim to have identified a new Earth-like planet, but the findings have not yet been independently verified or peer-reviewed.",
     0),
    ("Some experts question effectiveness of new policy, others support it",
     "The proposed regulation has drawn mixed reactions from economists, with some citing evidence of potential benefits while others warn of unintended consequences.",
     0),
]


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "n_samples": len(y_true),
    }


def run_benchmark():
    detector = FakeNewsDetector()
    y_true, y_pred, results = [], [], []

    for title, body, label in BENCHMARK_DATA:
        analysis = detector.analyze(title, body)
        # Convert verdict to binary
        pred = 1 if analysis["verdict"] in ("FAKE", "SUSPICIOUS") else 0
        y_true.append(label)
        y_pred.append(pred)
        results.append({
            "title": title[:60] + "...",
            "true_label": "FAKE" if label == 1 else "REAL",
            "predicted": analysis["verdict"],
            "fake_prob": analysis["fake_probability"],
            "correct": label == pred,
        })

    metrics = compute_metrics(y_true, y_pred)

    print("\n" + "="*65)
    print("  BENCHMARK EVALUATION REPORT")
    print("="*65)
    print(f"\n  Samples evaluated : {metrics['n_samples']}")
    print(f"  Accuracy          : {metrics['accuracy']*100:.1f}%")
    print(f"  Precision         : {metrics['precision']*100:.1f}%")
    print(f"  Recall            : {metrics['recall']*100:.1f}%")
    print(f"  F1 Score          : {metrics['f1_score']*100:.1f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    True Positives  : {metrics['true_positives']} (fake correctly flagged)")
    print(f"    True Negatives  : {metrics['true_negatives']} (real correctly passed)")
    print(f"    False Positives : {metrics['false_positives']} (real flagged as fake)")
    print(f"    False Negatives : {metrics['false_negatives']} (fake missed)")

    print("\n  Per-Sample Results:")
    print("  " + "-"*63)
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} [{r['true_label']:4s}→{r['predicted']:12s}] {r['fake_prob']:5.1f}%  {r['title']}")

    print("\n" + "="*65 + "\n")
    return metrics


if __name__ == "__main__":
    run_benchmark()
