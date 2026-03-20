import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re


def parse_cipher_blocks(file_path: Path) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    freq_rows: list[tuple[str, int, list[int]]] = []
    freq_pattern = re.compile(r"^([A-Z]{3}):\s*count=(\d+);\s*positions=\[(.*)\]\s*$")

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            freq_match = freq_pattern.match(line)
            if freq_match:
                trigram = freq_match.group(1)
                count = int(freq_match.group(2))
                positions_blob = freq_match.group(3).strip()
                pos_list: list[int] = []
                if positions_blob:
                    for token in positions_blob.split(","):
                        token = token.strip()
                        if token.isdigit():
                            pos_list.append(int(token))
                if len(pos_list) != count:
                    continue
                freq_rows.append((trigram, count, pos_list))
                continue

            if "=" not in line:
                continue

            lhs, rhs = line.split("=", 1)
            lhs = lhs.strip().upper()
            trigram = "".join(ch for ch in rhs.strip().upper() if "A" <= ch <= "Z")

            if not lhs.startswith("C_") or len(trigram) != 3:
                continue

            idx_text = lhs[2:].strip()
            if not idx_text.isdigit():
                continue

            blocks.append((int(idx_text), trigram))

    # Support files already in aggregated report format.
    if not blocks and freq_rows:
        for trigram, _, pos_list in freq_rows:
            for pos in pos_list:
                blocks.append((pos, trigram))
        blocks.sort(key=lambda x: x[0])

    return blocks


def build_report(blocks: list[tuple[int, str]], top_n: int | None) -> str:
    counts = Counter(trigram for _, trigram in blocks)
    positions: dict[str, list[int]] = defaultdict(list)
    for pos, trigram in blocks:
        positions[trigram].append(pos)

    rows = [(tri, counts[tri], positions[tri]) for tri in counts]
    rows.sort(key=lambda x: (-x[1], x[2][0], x[0]))

    if top_n is not None:
        rows = rows[:top_n]

    lines = [
        "MOST COMMON CIPHERTEXT TRIGRAMS",
        "=" * 80,
        f"Total blocks parsed: {len(blocks)}",
        f"Unique trigrams: {len(counts)}",
        "",
    ]

    for trigram, count, pos_list in rows:
        pos_text = ", ".join(str(p) for p in pos_list)
        lines.append(f"{trigram}: count={count}; positions=[{pos_text}]")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the most common 3-letter ciphertext blocks into a txt file."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="text_and_log_files/cipher_blocks.txt",
        help="Path to ciphertext block file (default: text_and_log_files/cipher_blocks.txt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="text_and_log_files/most_common_cipher_trigrams.txt",
        help="Output txt path (default: text_and_log_files/most_common_cipher_trigrams.txt)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Optional: keep only top N most common trigrams",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    blocks = parse_cipher_blocks(input_path)
    if not blocks:
        raise ValueError("No valid ciphertext blocks found in input file.")

    report = build_report(blocks, args.top)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
