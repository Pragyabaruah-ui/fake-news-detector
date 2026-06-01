"""
Fake News Detector — Interactive CLI
======================================
Run this file to analyze articles interactively.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.detector import FakeNewsDetector


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║          🔍  FAKE NEWS DETECTOR  v1.0                    ║
║     NLP-powered article authenticity classifier          ║
╚══════════════════════════════════════════════════════════╝
""")


def print_result(result: dict):
    verdict = result["verdict"]
    fake_pct = result["fake_probability"]
    real_pct = result["real_probability"]
    conf = result["confidence"]

    # Verdict color-coding via Unicode
    icons = {"FAKE": "🚨", "SUSPICIOUS": "⚠️ ", "LIKELY REAL": "✅"}
    icon = icons.get(verdict, "❓")

    print(f"\n{'─'*56}")
    print(f"  {icon}  VERDICT: {verdict}  ({conf} confidence)")
    print(f"{'─'*56}")
    print(f"  Fake probability : {fake_pct:>5.1f}%  {'█' * int(fake_pct/5)}")
    print(f"  Real probability : {real_pct:>5.1f}%  {'█' * int(real_pct/5)}")

    print(f"\n  Model breakdown:")
    scores = result["model_scores"]
    print(f"    Naive Bayes        : {scores['naive_bayes']}%")
    print(f"    Linguistic Rules   : {scores['linguistic_heuristic']}%")
    print(f"    Ensemble           : {scores['ensemble']}%")

    if result["red_flags"]:
        print(f"\n  🚩 Red Flags ({len(result['red_flags'])}):")
        for flag in result["red_flags"]:
            print(f"     • {flag}")

    if result["green_flags"]:
        print(f"\n  ✅ Credibility Signals ({len(result['green_flags'])}):")
        for flag in result["green_flags"]:
            print(f"     • {flag}")

    ling = result["linguistic_features"]
    print(f"\n  📊 Linguistic Analysis:")
    print(f"     Word count        : {ling['word_count']}")
    print(f"     Avg word length   : {ling['avg_word_length']:.2f} chars")
    print(f"     Lexical diversity : {ling['type_token_ratio']:.2%}")
    print(f"     Sensational score : {ling['sensational_score']}")
    print(f"     Credibility score : {ling['credibility_score']}")
    print(f"{'─'*56}\n")


def run_demo():
    """Run a quick demo with pre-set examples."""
    detector = FakeNewsDetector()

    demo_cases = [
        {
            "title": "Federal Reserve raises rates; economists cite inflation",
            "body": "The Federal Reserve raised interest rates by 25 basis points, according to bank officials. Economists cited data showing persistent inflation as the primary driver.",
        },
        {
            "title": "SHOCKING BOMBSHELL: They've been HIDING the CURE for cancer!!!",
            "body": "WAKE UP!! Big Pharma has been suppressing the 100% PROVEN natural cure for DECADES!! You won't believe this EXPOSED secret!! SHARE NOW before they delete it!!!",
        },
        {
            "title": "City launches new recycling program following council vote",
            "body": "The city council voted 7-2 to approve the expanded recycling program, which officials say will reduce landfill waste by an estimated 30 percent over five years.",
        },
    ]

    print("\n  Running demo analysis on 3 sample articles...\n")
    for i, case in enumerate(demo_cases, 1):
        print(f"  Article {i}: \"{case['title']}\"")
        result = detector.analyze(case["title"], case["body"])
        print_result(result)
        input("  Press Enter to continue...\n")


def run_interactive():
    """Interactive article analysis loop."""
    detector = FakeNewsDetector()

    print("  Enter article details to analyze. Type 'quit' to exit.\n")

    while True:
        print("  " + "─" * 54)
        title = input("  Article headline (or 'quit'): ").strip()
        if title.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye!\n")
            break
        if not title:
            continue

        body = input("  Article body (optional, press Enter to skip): ").strip()

        print("\n  Analyzing...")
        result = detector.analyze(title, body)
        print_result(result)


def main():
    print_banner()
    print("  Choose a mode:")
    print("  [1] Demo — analyze 3 sample articles")
    print("  [2] Interactive — analyze your own articles")
    print("  [3] Benchmark — run evaluation on test dataset")
    print("  [4] Exit\n")

    choice = input("  Enter choice (1-4): ").strip()

    if choice == "1":
        run_demo()
    elif choice == "2":
        run_interactive()
    elif choice == "3":
        from src.evaluate import run_benchmark
        run_benchmark()
    elif choice == "4":
        print("\n  Goodbye!\n")
    else:
        print("\n  Invalid choice. Running demo...\n")
        run_demo()


if __name__ == "__main__":
    main()
