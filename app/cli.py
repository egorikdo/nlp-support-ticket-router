"""Usage: python -m app.cli 'Не могу войти в личный кабинет'"""
import argparse
import json
from app.model import classify_text, evaluate


def main():
    parser = argparse.ArgumentParser(description="Classify a Russian support message")
    parser.add_argument("text", nargs="?", help="Text to classify")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on a held-out sample")
    args = parser.parse_args()
    if args.evaluate:
        print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
    elif args.text:
        print(json.dumps(classify_text(args.text), ensure_ascii=False, indent=2))
    else:
        parser.error("Provide a message or use --evaluate")

if __name__ == "__main__":
    main()
