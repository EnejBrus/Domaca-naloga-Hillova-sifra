import argparse
import time
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np


DEFAULT_ENGLISH_TRIGRAMS = [
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


COMMON_ENGLISH_WORDS = {
    "THE",
    "AND",
    "THAT",
    "HAVE",
    "FOR",
    "NOT",
    "WITH",
    "YOU",
    "THIS",
    "BUT",
    "HIS",
    "FROM",
    "THEY",
    "SAY",
    "HER",
    "SHE",
    "WILL",
    "ONE",
    "ALL",
    "WOULD",
    "THERE",
    "THEIR",
    "WHAT",
    "ABOUT",
    "WHICH",
    "WHEN",
    "MAKE",
    "CAN",
    "LIKE",
    "TIME",
    "JUST",
    "KNOW",
    "TAKE",
    "PEOPLE",
    "INTO",
    "YEAR",
    "YOUR",
    "GOOD",
    "SOME",
    "COULD",
    "THEM",
    "SEE",
    "OTHER",
    "THAN",
    "THEN",
    "NOW",
    "LOOK",
    "ONLY",
    "COME",
    "ITS",
    "OVER",
    "THINK",
    "ALSO",
    "BACK",
    "AFTER",
    "USE",
    "TWO",
    "HOW",
    "OUR",
    "WORK",
    "FIRST",
    "WELL",
    "WAY",
    "EVEN",
    "NEW",
    "WANT",
    "BECAUSE",
    "ANY",
    "THESE",
    "GIVE",
    "DAY",
    "MOST",
    "US",
    "IS",
    "ARE",
    "WAS",
    "WERE",
    "BE",
    "TO",
    "OF",
    "IN",
    "IT",
    "ON",
    "AS",
    "AT",
    "BY",
    "AN",
    "OR",
    "IF",
}


def clean_letters(text: str) -> str:
    return "".join(ch for ch in text.upper() if "A" <= ch <= "Z")


def parse_raw_ciphertext(path: Path, expected_blocks: int = 456) -> tuple[str, list[str]]:
    raw = path.read_text(encoding="utf-8")
    letters = clean_letters(raw)
    need = expected_blocks * 3
    if len(letters) < need:
        raise ValueError(f"Not enough letters: got {len(letters)}, need at least {need}.")
    letters = letters[:need]
    blocks = [letters[i : i + 3] for i in range(0, need, 3)]
    return letters, blocks


def trigram_to_vector(trigram: str) -> np.ndarray:
    return np.array([ord(ch) - ord("A") for ch in trigram], dtype=int)


def build_matrix_from_trigrams(t1: str, t2: str, t3: str) -> np.ndarray:
    return np.column_stack([trigram_to_vector(t1), trigram_to_vector(t2), trigram_to_vector(t3)])


def mod_inverse(a: int, m: int) -> int:
    a = a % m
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    raise ValueError(f"No modular inverse for {a} mod {m}")


def matrix_inverse_mod_3x3(matrix: np.ndarray, mod: int = 26) -> np.ndarray:
    det = int(round(np.linalg.det(matrix))) % mod
    if gcd(det, mod) != 1:
        raise ValueError("Matrix not invertible modulo 26")

    det_inv = mod_inverse(det, mod)
    m = matrix
    adj = np.array(
        [
            [m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1], m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2], m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]],
            [m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2], m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0], m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]],
            [m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0], m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1], m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]],
        ],
        dtype=int,
    )
    return (det_inv * adj) % mod


def decrypt_with_key(cipher_letters: str, key_matrix: np.ndarray) -> str:
    key_inv = matrix_inverse_mod_3x3(key_matrix, 26)
    nums = [ord(ch) - ord("A") for ch in cipher_letters]

    out = []
    for i in range(0, len(nums), 3):
        block = np.array(nums[i : i + 3], dtype=int)
        dec = (key_inv @ block) % 26
        out.extend(int(x) for x in dec)

    return "".join(chr(x + ord("A")) for x in out)


def english_word_score(text: str) -> tuple[int, list[tuple[str, int]]]:
    score = 0
    details = []
    for word in COMMON_ENGLISH_WORDS:
        count = text.count(word)
        if count > 0:
            score += count
            details.append((word, count))
    details.sort(key=lambda x: (x[1], len(x[0]), x[0]), reverse=True)
    return score, details


def trigram_score(text: str) -> tuple[int, int, int, int]:
    the_count = text.count("THE")
    and_count = text.count("AND")
    ing_count = text.count("ING")
    total = the_count + and_count + ing_count
    return total, the_count, and_count, ing_count


def format_matrix(matrix: np.ndarray) -> str:
    rows = ["[" + ", ".join(str(int(v)) for v in row) + "]" for row in matrix.tolist()]
    return "[" + ", ".join(rows) + "]"


def ordered_unique_blocks_by_frequency(blocks: list[str], top_n: int, fixed_c1: str) -> list[str]:
    counts = Counter(blocks)
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ordered = [tri for tri, _ in items]

    if fixed_c1 not in ordered:
        raise ValueError(f"Fixed ciphertext trigram {fixed_c1} not found in text.")

    selected = [fixed_c1]
    for tri in ordered:
        if tri == fixed_c1:
            continue
        selected.append(tri)
        if len(selected) >= top_n:
            break

    return selected


def search_keys(
    cipher_letters: str,
    cipher_blocks: list[str],
    fixed_c1: str,
    fixed_m1: str,
    top_cipher: int,
    english_trigrams: list[str],
    top_results: int,
) -> tuple[list[dict], dict]:
    c_candidates = ordered_unique_blocks_by_frequency(cipher_blocks, top_cipher, fixed_c1)

    m_candidates = [t for t in english_trigrams if t != fixed_m1]

    valid_m_pairs = []
    for m2 in m_candidates:
        for m3 in m_candidates:
            if m2 == m3:
                continue
            m_matrix = build_matrix_from_trigrams(fixed_m1, m2, m3)
            try:
                m_inv = matrix_inverse_mod_3x3(m_matrix, 26)
                valid_m_pairs.append((m2, m3, m_inv))
            except ValueError:
                continue

    results = []
    tested = 0
    invertible_k = 0

    for c2 in c_candidates[1:]:
        for c3 in c_candidates[1:]:
            if c2 == c3:
                continue
            c_matrix = build_matrix_from_trigrams(fixed_c1, c2, c3)

            for m2, m3, m_inv in valid_m_pairs:
                tested += 1
                k_matrix = (c_matrix @ m_inv) % 26
                try:
                    _ = matrix_inverse_mod_3x3(k_matrix, 26)
                except ValueError:
                    continue

                invertible_k += 1
                plaintext = decrypt_with_key(cipher_letters, k_matrix)
                tri_total, tri_the, tri_and, tri_ing = trigram_score(plaintext)
                eng_score, eng_details = english_word_score(plaintext)

                results.append(
                    {
                        "english_score": eng_score,
                        "trigram_score": tri_total,
                        "the": tri_the,
                        "and": tri_and,
                        "ing": tri_ing,
                        "c2": c2,
                        "c3": c3,
                        "m2": m2,
                        "m3": m3,
                        "k": k_matrix.copy(),
                        "plaintext": plaintext,
                        "english_details": eng_details,
                    }
                )

    results.sort(
        key=lambda r: (r["english_score"], r["trigram_score"], r["the"], r["and"], r["ing"]),
        reverse=True,
    )

    if top_results is not None:
        results = results[:top_results]

    stats = {
        "cipher_candidates": len(c_candidates),
        "valid_m_pairs": len(valid_m_pairs),
        "tested_combinations": tested,
        "invertible_k_found": invertible_k,
    }
    return results, stats


def write_report(path: Path, results: list[dict], stats: dict, elapsed_seconds: float, preview_len: int) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("YFY->THE TIMED KEY SEARCH RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Runtime seconds: {elapsed_seconds:.4f}\n")
        f.write(f"Cipher candidates used: {stats['cipher_candidates']}\n")
        f.write(f"Valid M pairs used: {stats['valid_m_pairs']}\n")
        f.write(f"Combinations tested: {stats['tested_combinations']}\n")
        f.write(f"Invertible K found: {stats['invertible_k_found']}\n")
        f.write(f"Results written: {len(results)}\n\n")

        for i, r in enumerate(results, start=1):
            f.write(f"Candidate #{i}\n")
            f.write(f"English score: {r['english_score']}\n")
            f.write(f"Trigram score (THE+AND+ING): {r['trigram_score']}\n")
            f.write(f"THE={r['the']}, AND={r['and']}, ING={r['ing']}\n")
            f.write(f"Mapping: C1=YFY,C2={r['c2']},C3={r['c3']} -> M1=THE,M2={r['m2']},M3={r['m3']}\n")
            f.write(f"K matrix: {format_matrix(r['k'])}\n")
            top_words = ", ".join(f"{w}:{c}" for w, c in r["english_details"][:12])
            f.write(f"Top word hits: {top_words}\n")
            f.write(f"Plaintext preview ({preview_len}): {r['plaintext'][:preview_len]}\n")
            f.write("-" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix YFY->THE, enumerate invertible K candidates, score plaintext, sort, and time the run."
    )
    parser.add_argument("--input", default="cipher_imput_raw.txt", help="Path to raw ciphertext file.")
    parser.add_argument(
        "--output",
        default="yfy_fixed_k_candidates_scored_timed.txt",
        help="Output txt file for sorted candidates.",
    )
    parser.add_argument("--expected-blocks", type=int, default=456, help="Expected number of 3-letter blocks.")
    parser.add_argument("--fixed-c1", default="YFY", help="Fixed ciphertext trigram C1.")
    parser.add_argument("--fixed-m1", default="THE", help="Fixed plaintext trigram M1.")
    parser.add_argument(
        "--top-cipher",
        type=int,
        default=20,
        help="Use top-N most frequent ciphertext trigrams as C-candidate pool (includes fixed C1).",
    )
    parser.add_argument(
        "--top-results",
        type=int,
        default=200,
        help="How many best scored results to write.",
    )
    parser.add_argument(
        "--preview-len",
        type=int,
        default=260,
        help="Plaintext preview length in report.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fixed_c1 = clean_letters(args.fixed_c1)
    fixed_m1 = clean_letters(args.fixed_m1)
    if len(fixed_c1) != 3 or len(fixed_m1) != 3:
        raise ValueError("fixed-c1 and fixed-m1 must each be exactly 3 letters.")

    cipher_letters, cipher_blocks = parse_raw_ciphertext(input_path, args.expected_blocks)

    start = time.perf_counter()
    results, stats = search_keys(
        cipher_letters=cipher_letters,
        cipher_blocks=cipher_blocks,
        fixed_c1=fixed_c1,
        fixed_m1=fixed_m1,
        top_cipher=args.top_cipher,
        english_trigrams=DEFAULT_ENGLISH_TRIGRAMS,
        top_results=args.top_results,
    )
    elapsed = time.perf_counter() - start

    output_path = Path(args.output)
    write_report(output_path, results, stats, elapsed, args.preview_len)

    print(f"Report written to: {output_path}")
    print(f"Runtime seconds: {elapsed:.4f}")
    if results:
        best = results[0]
        print(f"Best English score: {best['english_score']}")
        print(f"Best K: {format_matrix(best['k'])}")


if __name__ == "__main__":
    main()
