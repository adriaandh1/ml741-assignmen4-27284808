"""Commands for reproduction and saved-result checks."""

import argparse
import json

import experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["prepare", "som", "clustering", "assessment", "bootstrap", "verify", "smoke"],
    )
    args = parser.parse_args()
    if args.stage == "verify":
        from verification import verify

        result = verify()
    else:
        result = getattr(experiments, args.stage)()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
