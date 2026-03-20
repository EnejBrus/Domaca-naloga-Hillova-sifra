import argparse
from collections import Counter, defaultdict
from pathlib import Path


def clean_letters(text: str) -> str:
    return "".join(ch for ch in text.upper() if "A" <= ch <= "Z")


def split_blocks(text: str, block_size: int = 3) -> list[str]:
    usable = len(text) - (len(text) % block_size)
    return [text[i : i + block_size] for i in range(0, usable, block_size)]


def build_report(blocks: list[str], expected_blocks: int | None = None) -> str:
    counts = Counter(blocks)
    positions = defaultdict(list)
    for idx, block in enumerate(blocks, start=1):
        positions[block].append(idx)

    rows = [(tri, cnt, positions[tri]) for tri, cnt in counts.items()]
    rows.sort(key=lambda x: (-x[1], x[2][0], x[0]))

    lines = [
        "MOST COMMON CIPHERTEXT 3-LETTER BLOCKS",
        "=" * 80,
        f"Total blocks used: {len(blocks)}",
        f"Unique blocks: {len(counts)}",
    ]
    if expected_blocks is not None:
        lines.append(f"Expected blocks: {expected_blocks}")
    lines.append("")

    for trigram, count, pos_list in rows:
        pos_text = ", ".join(str(p) for p in pos_list)
        lines.append(f"{trigram}: count={count}; positions=[{pos_text}]")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count occurrences of ciphertext 3-letter blocks and save report to txt."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="cipher_imput_raw.txt",
        help="Path to raw ciphertext input file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="most_common_cipher_trigrams_456.txt",
        help="Path to output txt report.",
    )
    parser.add_argument(
        "--expected-blocks",
        type=int,
        default=456,
        help="Expected number of 3-letter blocks (default: 456).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_text = input_path.read_text(encoding="utf-8")
    clean = clean_letters(raw_text)

    required_letters = args.expected_blocks * 3 if args.expected_blocks else None
    if required_letters is not None:
        if len(clean) < required_letters:
            raise ValueError(
                f"Not enough ciphertext letters: got {len(clean)}, need {required_letters}."
            )
        if len(clean) > required_letters:
            clean = clean[:required_letters]

    blocks = split_blocks(clean, 3)

    if args.expected_blocks is not None and len(blocks) != args.expected_blocks:
        raise ValueError(
            f"Block count mismatch: got {len(blocks)}, expected {args.expected_blocks}."
        )

    report = build_report(blocks, args.expected_blocks)

    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
