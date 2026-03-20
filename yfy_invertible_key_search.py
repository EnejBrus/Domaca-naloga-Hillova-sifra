import argparse
import re
from collections import Counter
from itertools import permutations
from math import gcd
from pathlib import Path

import numpy as np


TOP20_ENGLISH_TRIGRAMS = [
    "THE",
    "AND",
    "ING",
    "HER",
    "ERE",
    "ENT",
    "THA",
    "NTH",
    "WAS",
    "ETH",
    "FOR",
    "DTH",
    "HAT",
    "ION",
    "TIO",
    "VER",
    "TER",
    "HES",
    "ATE",
    "ALL",
]


COMMON_WORDS = {
    "THE",
    "AND",
    "ING",
    "THAT",
    "HAVE",
    "FOR",
    "NOT",
    "WITH",
    "YOU",
    "THIS",
    "FROM",
    "THEY",
    "WILL",
    "ONE",
    "ALL",
    "THERE",
    "THEIR",
    "WHAT",
    "WHEN",
    "TIME",
    "JUST",
    "KNOW",
    "YOUR",
    "GOOD",
    "THAN",
    "THEN",
    "NOW",
    "LOOK",
    "COME",
    "OVER",
    "AFTER",
    "HOW",
    "OUR",
    "FIRST",
    "WAY",
    "WANT",
    "THESE",
    "MOST",
}


def trigram_to_vec(trigram: str) -> np.ndarray:
    return np.array([ord(ch) - ord("A") for ch in trigram], dtype=int)


def vec_to_trigram(vec: np.ndarray) -> str:
    return "".join(chr(int(v % 26) + ord("A")) for v in vec)


def matrix_from_trigrams(trigrams: tuple[str, str, str]) -> np.ndarray:
    return np.column_stack([trigram_to_vec(t) for t in trigrams])


def mod_inverse(a: int, m: int) -> int:
    def egcd(x: int, y: int) -> tuple[int, int, int]:
        if x == 0:
            return y, 0, 1
        g, x1, y1 = egcd(y % x, x)
        return g, y1 - (y // x) * x1, x1

    _, x, _ = egcd(a % m, m)
    return (x % m + m) % m


def inverse_mod_26_3x3(matrix: np.ndarray) -> np.ndarray:
    det = int(round(np.linalg.det(matrix))) % 26
    if gcd(det, 26) != 1:
        raise ValueError("Matrix is not invertible mod 26")

    m = matrix
    adj = np.array(
        [
            [m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1], m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2], m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]],
            [m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2], m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0], m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]],
            [m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0], m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1], m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]],
        ],
        dtype=int,
    )
    return (mod_inverse(det, 26) * adj) % 26


def parse_frequency_report(path: Path) -> list[tuple[str, int, list[int]]]:
    pattern = re.compile(r"^([A-Z]{3}):\s*count=(\d+);\s*positions=\[(.*)\]\s*$")
    rows: list[tuple[str, int, list[int]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            m = pattern.match(line)
            if not m:
                continue
            tri = m.group(1)
            count = int(m.group(2))
            pos_blob = m.group(3).strip()
            pos_list: list[int] = []
            if pos_blob:
                for token in pos_blob.split(","):
                    token = token.strip()
                    if token.isdigit():
                        pos_list.append(int(token))
            if len(pos_list) != count:
                continue
            rows.append((tri, count, pos_list))
    return rows


def format_matrix(matrix: np.ndarray) -> str:
    rows = ["[" + ", ".join(str(int(v)) for v in row) + "]" for row in matrix.tolist()]
    return "[" + ", ".join(rows) + "]"


def build_cipher_stream(rows: list[tuple[str, int, list[int]]]) -> tuple[str, int]:
    by_position: dict[int, str] = {}
    for trigram, _, positions in rows:
        for pos in positions:
            by_position[pos] = trigram

    if not by_position:
        return "", 0

    ordered_positions = sorted(by_position)
    ordered_blocks = [by_position[pos] for pos in ordered_positions]
    return "".join(ordered_blocks), len(ordered_blocks)


def decrypt_ciphertext(cipher_text: str, key_inv: np.ndarray) -> str:
    nums = [ord(ch) - ord("A") for ch in cipher_text]
    plaintext_nums: list[int] = []
    for i in range(0, len(nums), 3):
        block = np.array(nums[i : i + 3], dtype=int)
        dec = (key_inv @ block) % 26
        plaintext_nums.extend(int(v) for v in dec)
    return "".join(chr(v + ord("A")) for v in plaintext_nums)


def english_plaintext_score(plaintext: str) -> tuple[int, int, int, int]:
    the_count = plaintext.count("THE")
    and_count = plaintext.count("AND")
    ing_count = plaintext.count("ING")
    trigram_score = the_count + and_count + ing_count
    word_hits = sum(plaintext.count(word) for word in COMMON_WORDS)
    return word_hits, trigram_score, the_count, and_count + ing_count


def score_key_by_known_trigrams(
    k_inv: np.ndarray,
    known_cipher_rows: list[tuple[str, int, list[int]]],
    english_trigram_set: set[str],
) -> tuple[int, int, list[tuple[str, str, int]]]:
    weighted_hits = 0
    unique_hits = 0
    decoded: list[tuple[str, str, int]] = []

    for cipher_tri, count, _ in known_cipher_rows:
        plain_tri = vec_to_trigram((k_inv @ trigram_to_vec(cipher_tri)) % 26)
        decoded.append((cipher_tri, plain_tri, count))
        if plain_tri in english_trigram_set:
            weighted_hits += count
            unique_hits += 1

    return weighted_hits, unique_hits, decoded


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate invertible Hill key matrices with fixed YFY->THE and ordered mappings "
            "using C2,C3 from top ciphertext trigrams and M2,M3 from top English trigrams."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default="text_and_log_files/trigram_frequency_report.txt",
        help="Input trigram frequency report path",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="text_and_log_files/yfy_invertible_key_candidates.txt",
        help="Output txt report path",
    )
    parser.add_argument(
        "--fixed-cipher",
        default="YFY",
        help="Fixed ciphertext trigram (default: YFY)",
    )
    parser.add_argument(
        "--fixed-plain",
        default="THE",
        help="Fixed plaintext trigram (default: THE)",
    )
    parser.add_argument(
        "--top-cipher",
        type=int,
        default=20,
        help="How many top ciphertext trigrams to consider for C2/C3 (excluding fixed)",
    )
    parser.add_argument(
        "--top-output",
        type=int,
        default=250,
        help="How many best candidates to write",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    rows = parse_frequency_report(input_path)
    if not rows:
        raise ValueError("No trigram rows found in the input report.")

    fixed_cipher = args.fixed_cipher.strip().upper()
    fixed_plain = args.fixed_plain.strip().upper()
    if len(fixed_cipher) != 3 or len(fixed_plain) != 3:
        raise ValueError("Fixed trigrams must be 3 letters long.")

    counts = Counter({tri: cnt for tri, cnt, _ in rows})
    first_pos = {tri: pos_list[0] for tri, _, pos_list in rows if pos_list}
    sorted_cipher = sorted(
        [tri for tri in counts if tri != fixed_cipher],
        key=lambda t: (-counts[t], first_pos.get(t, 10**9), t),
    )
    cipher_pool = sorted_cipher[: max(2, args.top_cipher)]

    plain_pool = [t for t in TOP20_ENGLISH_TRIGRAMS if t != fixed_plain]
    english_set = set(TOP20_ENGLISH_TRIGRAMS)
    cipher_stream, sampled_blocks = build_cipher_stream(rows)
    if sampled_blocks == 0:
        raise ValueError("Cannot reconstruct ciphertext sample from frequency rows.")

    total_attempts = 0
    non_invertible_m = 0
    non_invertible_k = 0
    kept = []

    for c2, c3 in permutations(cipher_pool, 2):
        c_triplet = (fixed_cipher, c2, c3)
        c_matrix = matrix_from_trigrams(c_triplet)

        for m2, m3 in permutations(plain_pool, 2):
            total_attempts += 1
            m_triplet = (fixed_plain, m2, m3)
            m_matrix = matrix_from_trigrams(m_triplet)

            try:
                m_inv = inverse_mod_26_3x3(m_matrix)
            except ValueError:
                non_invertible_m += 1
                continue

            k = (c_matrix @ m_inv) % 26
            det_k = int(round(np.linalg.det(k))) % 26
            if gcd(det_k, 26) != 1:
                non_invertible_k += 1
                continue

            k_inv = inverse_mod_26_3x3(k)
            weighted_hits, unique_hits, decoded_rows = score_key_by_known_trigrams(
                k_inv=k_inv,
                known_cipher_rows=rows,
                english_trigram_set=english_set,
            )

            plaintext = decrypt_ciphertext(cipher_stream, k_inv)
            word_hits, trigram_score, the_count, and_ing_sum = english_plaintext_score(plaintext)

            kept.append(
                {
                    "c_triplet": c_triplet,
                    "m_triplet": m_triplet,
                    "k": k,
                    "det_k": det_k,
                    "weighted_hits": weighted_hits,
                    "unique_hits": unique_hits,
                    "word_hits": word_hits,
                    "trigram_score": trigram_score,
                    "the_count": the_count,
                    "and_ing_sum": and_ing_sum,
                    "plaintext": plaintext,
                    "decoded_rows": decoded_rows,
                }
            )

    kept.sort(
        key=lambda r: (
            r["word_hits"],
            r["trigram_score"],
            r["the_count"],
            r["and_ing_sum"],
            r["weighted_hits"],
            r["unique_hits"],
            -sum(abs(int(x)) for x in r["k"].flatten()),
        ),
        reverse=True,
    )

    output_limit = min(args.top_output, len(kept))
    output_lines = [
        "INVERTIBLE KEY SEARCH (YFY -> THE)",
        "=" * 90,
        f"Input: {input_path}",
        f"Fixed mapping: {fixed_cipher} -> {fixed_plain}",
        f"Cipher sample blocks used for decryption score: {sampled_blocks}",
        f"Cipher pool size for C2/C3: {len(cipher_pool)}",
        f"Plain pool size for M2/M3: {len(plain_pool)}",
        f"Total attempts: {total_attempts}",
        f"Skipped (M not invertible): {non_invertible_m}",
        f"Skipped (K not invertible): {non_invertible_k}",
        f"Invertible K candidates: {len(kept)}",
        f"Written candidates: {output_limit}",
        "",
        "Sort key: word_hits desc, trigram_score desc, THE desc, (AND+ING) desc, weighted_hits desc",
        "weighted_hits = sum(count(cipher_trigram)) for decoded trigrams in top20 English set",
        "word_hits = count matches against a common-English word list in decrypted plaintext",
        "",
    ]

    for idx, row in enumerate(kept[:output_limit], start=1):
        output_lines.append(f"Candidate #{idx}")
        output_lines.append(
            f"Mappings: {row['c_triplet'][0]}->{row['m_triplet'][0]}, {row['c_triplet'][1]}->{row['m_triplet'][1]}, {row['c_triplet'][2]}->{row['m_triplet'][2]}"
        )
        output_lines.append(f"det(K) mod 26: {row['det_k']}")
        output_lines.append(
            f"English score: word_hits={row['word_hits']}, trigram_score={row['trigram_score']}, THE={row['the_count']}, AND+ING={row['and_ing_sum']}"
        )
        output_lines.append(
            f"Secondary score: weighted_hits={row['weighted_hits']}, unique_hits={row['unique_hits']}"
        )
        output_lines.append(f"K = {format_matrix(row['k'])}")

        top_decoded = sorted(row["decoded_rows"], key=lambda x: (-x[2], x[0]))[:12]
        decoded_text = ", ".join(f"{c}->{p}({cnt})" for c, p, cnt in top_decoded)
        output_lines.append(f"Top decoded trigrams: {decoded_text}")
        output_lines.append(f"Plaintext preview: {row['plaintext'][:300]}")
        output_lines.append("-" * 90)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    print(f"Invertible K candidates found: {len(kept)}")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
