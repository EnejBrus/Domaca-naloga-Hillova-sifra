import numpy as np
from math import gcd
from itertools import permutations
from pathlib import Path
import argparse
import time


DEFAULT_TOP_TRIGRAMS = [
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
    "ING",
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



class HillCipher:
    """Hill cipher implementation for encryption and decryption."""

    def __init__(self, key_matrix):
        """Initialize with a square key matrix."""
        self.key_matrix = np.array(key_matrix, dtype=int)
        if self.key_matrix.ndim != 2 or self.key_matrix.shape[0] != self.key_matrix.shape[1]:
            raise ValueError("Key matrix must be square.")
        self.n = self.key_matrix.shape[0]
        self.mod = 26

    def _pad_to_block_size(self, numbers):
        """Pad numeric text with X (23) so length is divisible by block size."""
        remainder = len(numbers) % self.n
        if remainder == 0:
            return numbers
        pad_len = self.n - remainder
        return numbers + [23] * pad_len

    def _mod_inverse(self, a, m):
        """Compute modular inverse of a modulo m."""

        def extended_gcd(x, y):
            if x == 0:
                return y, 0, 1
            gcd_val, x1, y1 = extended_gcd(y % x, x)
            return gcd_val, y1 - (y // x) * x1, x1

        _, x, _ = extended_gcd(a % m, m)
        return (x % m + m) % m

    def _matrix_inverse_mod(self, matrix, mod):
        """Compute inverse matrix modulo mod."""
        det = int(round(np.linalg.det(matrix))) % mod
        if gcd(det, mod) != 1:
            raise ValueError(f"Determinant {det} is not invertible modulo {mod}")

        det_inv = self._mod_inverse(det, mod)
        adj = self._adjugate_3x3(matrix) if self.n == 3 else self._adjugate_nxn(matrix)
        return (det_inv * adj) % mod

    def _adjugate_3x3(self, matrix):
        """Compute adjugate for a 3x3 matrix."""
        m = matrix
        return np.array(
            [
                [m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1], m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2], m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]],
                [m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2], m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0], m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]],
                [m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0], m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1], m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]],
            ],
            dtype=int,
        )

    def _adjugate_nxn(self, matrix):
        """Compute adjugate for an NxN matrix."""
        n = matrix.shape[0]
        adj = np.zeros_like(matrix, dtype=int)
        for i in range(n):
            for j in range(n):
                minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
                det_minor = int(round(np.linalg.det(minor)))
                adj[j, i] = ((-1) ** (i + j)) * det_minor
        return adj

    def text_to_numbers(self, text):
        """Convert text to numbers (A=0, ..., Z=25)."""
        return [ord(char.upper()) - ord("A") for char in text]

    def numbers_to_text(self, numbers):
        """Convert numbers back to text."""
        return "".join(chr((num % 26) + ord("A")) for num in numbers)

    def decrypt(self, ciphertext):
        """Decrypt ciphertext with the key matrix."""
        key_inv = self._matrix_inverse_mod(self.key_matrix, self.mod)
        cipher_numbers = self._pad_to_block_size(self.text_to_numbers(ciphertext))

        plaintext_numbers = []
        for i in range(0, len(cipher_numbers), self.n):
            block = np.array(cipher_numbers[i : i + self.n], dtype=int)
            decrypted_block = (key_inv @ block) % self.mod
            plaintext_numbers.extend(decrypted_block)

        return self.numbers_to_text(plaintext_numbers)

    def encrypt(self, plaintext):
        """Encrypt plaintext with the key matrix."""
        plain_numbers = self._pad_to_block_size(self.text_to_numbers(plaintext))

        ciphertext_numbers = []
        for i in range(0, len(plain_numbers), self.n):
            block = np.array(plain_numbers[i : i + self.n], dtype=int)
            encrypted_block = (self.key_matrix @ block) % self.mod
            ciphertext_numbers.extend(encrypted_block)

        return self.numbers_to_text(ciphertext_numbers)


def parse_cipher_blocks(file_path):
    """Parse lines like C_1 = ABC into an ordered list of trigram strings."""
    blocks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            _, rhs = line.split("=", 1)
            trigram = "".join(ch for ch in rhs.strip().upper() if "A" <= ch <= "Z")
            if len(trigram) == 3:
                blocks.append(trigram)
    return blocks


def parse_cipher_text_file(file_path):
    """Parse raw ciphertext text and keep only uppercase A-Z letters."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()
    return "".join(ch for ch in raw.upper() if "A" <= ch <= "Z")


def ciphertext_stream_from_blocks(cipher_blocks):
    """Build the full ciphertext stream by concatenating listed 3-letter blocks."""
    return "".join(cipher_blocks)


def consecutive_trigrams(text):
    """Return all overlapping 3-letter windows from text."""
    clean = "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
    if len(clean) < 3:
        return []
    return [clean[i : i + 3] for i in range(len(clean) - 2)]


def trigram_to_vector(trigram):
    """Map trigram ABC to a numeric vector [0, 1, 2]."""
    return np.array([ord(ch) - ord("A") for ch in trigram], dtype=int)


def build_matrix_from_trigrams(trigrams):
    """Build a 3x3 matrix where each trigram vector is a column."""
    columns = [trigram_to_vector(t) for t in trigrams]
    return np.column_stack(columns)


def format_matrix(matrix):
    """Convert numpy matrix into a compact string representation."""
    rows = ["[" + ", ".join(str(int(v)) for v in row) + "]" for row in matrix.tolist()]
    return "[" + ", ".join(rows) + "]"


def recover_key_from_triplets(cipher_triplet, plain_triplet):
    """Recover candidate Hill key: K = C * M^{-1} (mod 26)."""
    helper = HillCipher(np.eye(3, dtype=int))
    c_matrix = build_matrix_from_trigrams(cipher_triplet)
    m_matrix = build_matrix_from_trigrams(plain_triplet)
    m_inv = helper._matrix_inverse_mod(m_matrix, 26)
    return (c_matrix @ m_inv) % 26


def score_plaintext(plaintext):
    """Score plaintext by occurrences of the target English trigrams."""
    the_count = plaintext.count("THE")
    and_count = plaintext.count("AND")
    ing_count = plaintext.count("ING")
    total = the_count + and_count + ing_count
    return total, the_count, and_count, ing_count


def english_word_score(plaintext, min_word_len=3):
    """Score plaintext by counting dictionary words found as substrings."""
    text = "".join(ch for ch in plaintext.upper() if "A" <= ch <= "Z")
    words = [w for w in COMMON_ENGLISH_WORDS if len(w) >= min_word_len]

    score = 0
    details = []
    for word in words:
        count = text.count(word)
        if count > 0:
            score += count
            details.append((word, count))

    details.sort(key=lambda x: (x[1], len(x[0]), x[0]), reverse=True)
    return score, details


def parse_plain_trigrams_arg(plain_trigrams_arg):
    """Parse comma-separated plaintext trigrams and validate 3-letter format."""
    trigram_list = []
    for token in plain_trigrams_arg.split(","):
        t = "".join(ch for ch in token.strip().upper() if "A" <= ch <= "Z")
        if len(t) != 3:
            raise ValueError(f"Invalid trigram '{token}'. Each trigram must have exactly 3 letters.")
        trigram_list.append(t)
    if len(trigram_list) < 3:
        raise ValueError("Need at least 3 plaintext trigrams.")
    return trigram_list


def collect_valid_plain_options(target_plain_trigrams):
    """Return valid plaintext triplets and inverses where M is invertible mod 26."""
    helper = HillCipher(np.eye(3, dtype=int))
    valid_plain_options = []
    total = 0
    for plain_triplet in permutations(target_plain_trigrams, 3):
        total += 1
        m_matrix = build_matrix_from_trigrams(plain_triplet)
        try:
            m_inv = helper._matrix_inverse_mod(m_matrix, 26)
            valid_plain_options.append((plain_triplet, m_inv))
        except ValueError:
            continue
    return total, valid_plain_options


def estimate_runtime_seconds(c_permutations_benchmark, benchmark_seconds, total_c_permutations):
    """Estimate runtime in seconds from a benchmark over C-permutations."""
    if c_permutations_benchmark <= 0 or benchmark_seconds <= 0:
        return None
    per_c = benchmark_seconds / c_permutations_benchmark
    return per_c * total_c_permutations


def format_duration(seconds):
    """Return human-readable duration string from seconds."""
    if seconds is None:
        return "N/A"
    s = int(round(seconds))
    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{days}d {hours}h {minutes}m {secs}s"


def write_progress_file(
    progress_file,
    attempted_combinations,
    total_combinations,
    processed_c,
    total_c,
    start_time,
):
    """Write current brute-force progress to a text file."""
    if progress_file is None:
        return

    elapsed = time.perf_counter() - start_time if start_time is not None else 0.0
    pct = (100.0 * attempted_combinations / total_combinations) if total_combinations > 0 else 0.0
    rate = (attempted_combinations / elapsed) if elapsed > 0 else 0.0
    remaining = (total_combinations - attempted_combinations) if total_combinations >= attempted_combinations else 0
    eta_seconds = (remaining / rate) if rate > 0 else None

    with open(progress_file, "w", encoding="utf-8") as f:
        f.write("BRUTE FORCE PROGRESS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Attempted combinations: {attempted_combinations}\n")
        f.write(f"Total combinations: {total_combinations}\n")
        f.write(f"Progress: {pct:.4f}%\n")
        f.write(f"Processed C permutations: {processed_c}/{total_c}\n")
        f.write(f"Elapsed: {elapsed:.2f}s ({format_duration(elapsed)})\n")
        f.write(f"Rate: {rate:.2f} combinations/s\n")
        f.write(f"ETA: {format_duration(eta_seconds)}\n")


def brute_force_hill_keys(
    cipher_text,
    c_trigram_candidates,
    target_plain_trigrams=("THE", "AND", "ING"),
    min_score=1,
    limit=None,
    valid_plain_options=None,
    progress_file=None,
    total_combinations=None,
    total_c_permutations=None,
    progress_every_c=100,
    progress_start_time=None,
):
    """Try all ordered C-triplets with all permutations of target plaintext trigrams."""
    results = []

    if valid_plain_options is None:
        _, valid_plain_options = collect_valid_plain_options(target_plain_trigrams)

    valid_m = len(valid_plain_options)
    if total_c_permutations is None:
        n_c = len(c_trigram_candidates)
        total_c_permutations = n_c * (n_c - 1) * (n_c - 2)
    if total_combinations is None:
        total_combinations = total_c_permutations * valid_m

    processed = 0
    for c_indices in permutations(range(len(c_trigram_candidates)), 3):
        cipher_triplet = (
            c_trigram_candidates[c_indices[0]],
            c_trigram_candidates[c_indices[1]],
            c_trigram_candidates[c_indices[2]],
        )
        c_matrix = build_matrix_from_trigrams(cipher_triplet)

        for m_idx, (plain_triplet, m_inv) in enumerate(valid_plain_options, start=1):
            key = (c_matrix @ m_inv) % 26
            try:
                cipher = HillCipher(key)
                plaintext = cipher.decrypt(cipher_text)
            except ValueError:
                continue

            score, the_count, and_count, ing_count = score_plaintext(plaintext)
            if score >= min_score:
                eng_score, eng_details = english_word_score(plaintext)
                results.append(
                    {
                        "score": score,
                        "the": the_count,
                        "and": and_count,
                        "ing": ing_count,
                        "english_score": eng_score,
                        "english_details": eng_details,
                        "cipher_indices": (c_indices[0], c_indices[1], c_indices[2]),
                        "cipher_triplet": cipher_triplet,
                        "plain_triplet": plain_triplet,
                        "key": key.copy(),
                        "plaintext": plaintext,
                    }
                )

        processed += 1
        if progress_file is not None and (processed % max(1, progress_every_c) == 0):
            write_progress_file(
                progress_file=progress_file,
                attempted_combinations=processed * valid_m,
                total_combinations=total_combinations,
                processed_c=processed,
                total_c=total_c_permutations,
                start_time=progress_start_time,
            )
        if limit is not None and processed >= limit:
            break

    if progress_file is not None:
        write_progress_file(
            progress_file=progress_file,
            attempted_combinations=processed * valid_m,
            total_combinations=total_combinations,
            processed_c=processed,
            total_c=total_c_permutations,
            start_time=progress_start_time,
        )

    results.sort(key=lambda r: (r["score"], r["the"], r["and"], r["ing"]), reverse=True)
    return results


def write_results(results, output_path, preview_len=240):
    """Write scored brute-force results to disk in descending score order."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("HILL CIPHER BRUTE-FORCE RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total candidates written: {len(results)}\n\n")

        for idx, item in enumerate(results, start=1):
            f.write(f"Candidate #{idx}\n")
            f.write(f"Score(THE+AND+ING): {item['score']}\n")
            f.write(f"THE={item['the']}, AND={item['and']}, ING={item['ing']}\n")
            f.write(f"English word score: {item['english_score']}\n")
            f.write(
                "Cipher windows used: "
                f"W_{item['cipher_indices'][0]}={item['cipher_triplet'][0]}, "
                f"W_{item['cipher_indices'][1]}={item['cipher_triplet'][1]}, "
                f"W_{item['cipher_indices'][2]}={item['cipher_triplet'][2]}\n"
            )
            f.write(f"Mapped plaintext trigrams: {item['plain_triplet']}\n")
            f.write(f"Recovered key K: {format_matrix(item['key'])}\n")
            top_words = ", ".join(f"{w}:{c}" for w, c in item["english_details"][:10])
            f.write(f"Top matched words: {top_words}\n")
            f.write(f"Plaintext preview ({preview_len} chars): {item['plaintext'][:preview_len]}\n")
            f.write("-" * 80 + "\n")


def write_best_english_candidate(results, output_path, preview_len=800):
    """Write the single candidate with highest English word score."""
    if not results:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("No candidates found.\n")
        return None

    best = max(
        results,
        key=lambda r: (r["english_score"], r["score"], r["the"], r["and"], r["ing"]),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("BEST ENGLISH CANDIDATE\n")
        f.write("=" * 80 + "\n")
        f.write(f"English word score: {best['english_score']}\n")
        f.write(f"Trigram score(THE+AND+ING): {best['score']}\n")
        f.write(f"THE={best['the']}, AND={best['and']}, ING={best['ing']}\n")
        f.write(
            "Cipher windows used: "
            f"W_{best['cipher_indices'][0]}={best['cipher_triplet'][0]}, "
            f"W_{best['cipher_indices'][1]}={best['cipher_triplet'][1]}, "
            f"W_{best['cipher_indices'][2]}={best['cipher_triplet'][2]}\n"
        )
        f.write(f"Mapped plaintext trigrams: {best['plain_triplet']}\n")
        f.write(f"Recovered key K: {format_matrix(best['key'])}\n")
        top_words = ", ".join(f"{w}:{c}" for w, c in best["english_details"][:25])
        f.write(f"Top matched words: {top_words}\n")
        f.write("\n")
        f.write("Full plaintext:\n")
        f.write(best["plaintext"][:preview_len] + "\n")

    return best


def write_single_best_candidate(best, output_path, preview_len=800):
    """Write one already-selected best candidate dictionary to disk."""
    if best is None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("No candidates found.\n")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("BEST ENGLISH CANDIDATE\n")
        f.write("=" * 80 + "\n")
        f.write(f"English word score: {best['english_score']}\n")
        f.write(f"Trigram score(THE+AND+ING): {best['score']}\n")
        f.write(f"THE={best['the']}, AND={best['and']}, ING={best['ing']}\n")
        f.write(
            "Cipher windows used: "
            f"W_{best['cipher_indices'][0]}={best['cipher_triplet'][0]}, "
            f"W_{best['cipher_indices'][1]}={best['cipher_triplet'][1]}, "
            f"W_{best['cipher_indices'][2]}={best['cipher_triplet'][2]}\n"
        )
        f.write(f"Mapped plaintext trigrams: {best['plain_triplet']}\n")
        f.write(f"Recovered key K: {format_matrix(best['key'])}\n")
        top_words = ", ".join(f"{w}:{c}" for w, c in best["english_details"][:25])
        f.write(f"Top matched words: {top_words}\n")
        f.write("\n")
        f.write("Full plaintext:\n")
        f.write(best["plaintext"][:preview_len] + "\n")


def append_best_event(best_event_file, best, attempted_combinations, total_combinations, elapsed_seconds):
    """Append a timestamped log line when a new best candidate appears."""
    if best_event_file is None or best is None:
        return

    pct = (100.0 * attempted_combinations / total_combinations) if total_combinations > 0 else 0.0
    with open(best_event_file, "a", encoding="utf-8") as f:
        f.write(
            "NEW BEST | "
            f"attempted={attempted_combinations}/{total_combinations} ({pct:.4f}%) | "
            f"elapsed={elapsed_seconds:.2f}s | "
            f"eng={best['english_score']} | "
            f"tri={best['score']} | "
            f"windows=W_{best['cipher_indices'][0]},W_{best['cipher_indices'][1]},W_{best['cipher_indices'][2]} | "
            f"M={best['plain_triplet']}\n"
        )


def brute_force_best_only(
    cipher_text,
    c_trigram_candidates,
    target_plain_trigrams=("THE", "AND", "ING"),
    min_score=0,
    limit=None,
    valid_plain_options=None,
    progress_file=None,
    total_combinations=None,
    total_c_permutations=None,
    progress_every_c=100,
    progress_start_time=None,
    best_live_file=None,
    best_event_file=None,
):
    """Evaluate all combinations but keep only the best-scoring English candidate."""
    if valid_plain_options is None:
        _, valid_plain_options = collect_valid_plain_options(target_plain_trigrams)

    valid_m = len(valid_plain_options)
    if total_c_permutations is None:
        n_c = len(c_trigram_candidates)
        total_c_permutations = n_c * (n_c - 1) * (n_c - 2)
    if total_combinations is None:
        total_combinations = total_c_permutations * valid_m

    best = None
    processed = 0

    for c_indices in permutations(range(len(c_trigram_candidates)), 3):
        cipher_triplet = (
            c_trigram_candidates[c_indices[0]],
            c_trigram_candidates[c_indices[1]],
            c_trigram_candidates[c_indices[2]],
        )
        c_matrix = build_matrix_from_trigrams(cipher_triplet)

        for m_idx, (plain_triplet, m_inv) in enumerate(valid_plain_options, start=1):
            key = (c_matrix @ m_inv) % 26
            try:
                cipher = HillCipher(key)
                plaintext = cipher.decrypt(cipher_text)
            except ValueError:
                continue

            score, the_count, and_count, ing_count = score_plaintext(plaintext)
            if score < min_score:
                continue

            eng_score, eng_details = english_word_score(plaintext)
            candidate = {
                "score": score,
                "the": the_count,
                "and": and_count,
                "ing": ing_count,
                "english_score": eng_score,
                "english_details": eng_details,
                "cipher_indices": (c_indices[0], c_indices[1], c_indices[2]),
                "cipher_triplet": cipher_triplet,
                "plain_triplet": plain_triplet,
                "key": key.copy(),
                "plaintext": plaintext,
            }

            if best is None:
                best = candidate
                attempted = (processed * valid_m) + m_idx
                elapsed = (time.perf_counter() - progress_start_time) if progress_start_time is not None else 0.0
                write_single_best_candidate(best, best_live_file) if best_live_file is not None else None
                append_best_event(best_event_file, best, attempted, total_combinations, elapsed)
            else:
                best_key = (best["english_score"], best["score"], best["the"], best["and"], best["ing"])
                cand_key = (eng_score, score, the_count, and_count, ing_count)
                if cand_key > best_key:
                    best = candidate
                    attempted = (processed * valid_m) + m_idx
                    elapsed = (time.perf_counter() - progress_start_time) if progress_start_time is not None else 0.0
                    write_single_best_candidate(best, best_live_file) if best_live_file is not None else None
                    append_best_event(best_event_file, best, attempted, total_combinations, elapsed)

        processed += 1
        if progress_file is not None and (processed % max(1, progress_every_c) == 0):
            write_progress_file(
                progress_file=progress_file,
                attempted_combinations=processed * valid_m,
                total_combinations=total_combinations,
                processed_c=processed,
                total_c=total_c_permutations,
                start_time=progress_start_time,
            )
        if limit is not None and processed >= limit:
            break

    if progress_file is not None:
        write_progress_file(
            progress_file=progress_file,
            attempted_combinations=processed * valid_m,
            total_combinations=total_combinations,
            processed_c=processed,
            total_c=total_c_permutations,
            start_time=progress_start_time,
        )

    return best


def run_trigram_attack(
    cipher_blocks_path,
    output_path,
    min_score=1,
    limit=None,
    cipher_prefix_len=None,
    best_output_path=None,
    target_plain_trigrams=("THE", "AND", "ING"),
    count_only=False,
    benchmark_c_permutations=None,
    best_only=False,
    progress_file=None,
    progress_every_c=100,
    best_event_file=None,
    cipher_text_file=None,
):
    """Execute full trigram-based brute force and write sorted results."""
    if cipher_text_file is not None:
        cipher_text = parse_cipher_text_file(cipher_text_file)
        if len(cipher_text) < 3:
            raise ValueError("Need at least 3 ciphertext letters in --cipher-text-file.")
    else:
        cipher_blocks = parse_cipher_blocks(cipher_blocks_path)
        if len(cipher_blocks) < 3:
            raise ValueError("Need at least 3 ciphertext blocks to run the attack.")
        cipher_text = ciphertext_stream_from_blocks(cipher_blocks)

    if cipher_prefix_len is not None:
        if cipher_prefix_len < 3:
            raise ValueError("cipher_prefix_len must be at least 3.")
        cipher_text = cipher_text[:cipher_prefix_len]

    c_trigram_candidates = consecutive_trigrams(cipher_text)
    if len(c_trigram_candidates) < 3:
        raise ValueError("Need at least 3 consecutive trigrams from ciphertext.")

    total_m, valid_plain_options = collect_valid_plain_options(target_plain_trigrams)
    valid_m = len(valid_plain_options)
    n_c = len(c_trigram_candidates)
    total_c = n_c * (n_c - 1) * (n_c - 2)
    total_combinations = total_c * valid_m

    if count_only:
        print(f"Cipher letters used: {len(cipher_text)}")
        print(f"Cipher windows (consecutive trigrams): {n_c}")
        print(f"C permutations: {total_c}")
        print(f"M permutations total: {total_m}")
        print(f"M permutations valid: {valid_m}")
        print(f"Total combinations (C * valid M): {total_combinations}")
        return 0

    if benchmark_c_permutations is not None and benchmark_c_permutations > 0:
        bench_start = time.perf_counter()
        brute_force_hill_keys(
            cipher_text=cipher_text,
            c_trigram_candidates=c_trigram_candidates,
            target_plain_trigrams=target_plain_trigrams,
            min_score=min_score,
            limit=benchmark_c_permutations,
            valid_plain_options=valid_plain_options,
        )
        bench_elapsed = time.perf_counter() - bench_start
        expected_total_seconds = estimate_runtime_seconds(
            c_permutations_benchmark=benchmark_c_permutations,
            benchmark_seconds=bench_elapsed,
            total_c_permutations=total_c,
        )
        print(f"Benchmark C permutations: {benchmark_c_permutations}")
        print(f"Benchmark elapsed: {bench_elapsed:.2f}s")
        print(f"Estimated full runtime: {format_duration(expected_total_seconds)}")

    run_start = time.perf_counter()
    max_c_to_run = total_c if limit is None else min(limit, total_c)
    total_combinations_for_run = max_c_to_run * valid_m

    if progress_file is not None:
        write_progress_file(
            progress_file=progress_file,
            attempted_combinations=0,
            total_combinations=total_combinations_for_run,
            processed_c=0,
            total_c=max_c_to_run,
            start_time=run_start,
        )

    if best_only:
        if best_event_file is not None:
            with open(best_event_file, "w", encoding="utf-8") as f:
                f.write("BEST-CANDIDATE EVENTS\n")
                f.write("=" * 80 + "\n")

        best = brute_force_best_only(
            cipher_text=cipher_text,
            c_trigram_candidates=c_trigram_candidates,
            target_plain_trigrams=target_plain_trigrams,
            min_score=min_score,
            limit=limit,
            valid_plain_options=valid_plain_options,
            progress_file=progress_file,
            total_combinations=total_combinations_for_run,
            total_c_permutations=max_c_to_run,
            progress_every_c=progress_every_c,
            progress_start_time=run_start,
            best_live_file=best_output_path if best_output_path is not None else output_path,
            best_event_file=best_event_file,
        )
        elapsed = time.perf_counter() - run_start
        target_best_file = best_output_path if best_output_path is not None else output_path
        write_single_best_candidate(best, target_best_file)
        print(f"Elapsed runtime: {elapsed:.2f}s ({format_duration(elapsed)})")
        return 1 if best is not None else 0

    results = brute_force_hill_keys(
        cipher_text=cipher_text,
        c_trigram_candidates=c_trigram_candidates,
        target_plain_trigrams=target_plain_trigrams,
        min_score=min_score,
        limit=limit,
        valid_plain_options=valid_plain_options,
        progress_file=progress_file,
        total_combinations=total_combinations_for_run,
        total_c_permutations=max_c_to_run,
        progress_every_c=progress_every_c,
        progress_start_time=run_start,
    )
    elapsed = time.perf_counter() - run_start
    write_results(results, output_path)
    if best_output_path is not None:
        write_best_english_candidate(results, best_output_path)
    print(f"Elapsed runtime: {elapsed:.2f}s ({format_duration(elapsed)})")
    return len(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brute-force 3x3 Hill key using ciphertext and plaintext trigram permutations."
    )
    parser.add_argument(
        "--cipher-blocks",
        default="cipher_blocks.txt",
        help="Path to ciphertext blocks file (default: cipher_blocks.txt)",
    )
    parser.add_argument(
        "--cipher-text-file",
        default=None,
        help="Optional path to raw ciphertext text file (A-Z letters are extracted and used directly).",
    )
    parser.add_argument(
        "--output",
        default="trigram_attack_results.txt",
        help="Output file for sorted decryption candidates.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=1,
        help="Only keep candidates with THE+AND+ING count >= min-score.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of ciphertext 3-permutations tested.",
    )
    parser.add_argument(
        "--cipher-prefix-len",
        type=int,
        default=None,
        help="Optional number of initial ciphertext letters to use (e.g., 60).",
    )
    parser.add_argument(
        "--best-output",
        default=None,
        help="Optional file path for the single best English-word candidate.",
    )
    parser.add_argument(
        "--plain-trigrams",
        default="THE,AND,ING",
        help="Comma-separated plaintext trigram pool for M permutations.",
    )
    parser.add_argument(
        "--use-top20-trigrams",
        action="store_true",
        help="Use the fixed top-20 English trigram list for M permutations.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only print combination counts, do not run decryption search.",
    )
    parser.add_argument(
        "--benchmark-c-permutations",
        type=int,
        default=None,
        help="Optional benchmark size (in C permutations) to estimate full runtime before the real run.",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Evaluate all combinations but keep/write only the best English candidate.",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Optional file path to continuously write attempted-combination progress.",
    )
    parser.add_argument(
        "--progress-every-c",
        type=int,
        default=100,
        help="How often to update progress file (in processed C permutations).",
    )
    parser.add_argument(
        "--best-event-file",
        default=None,
        help="Optional file that appends a line whenever a new best English candidate is found.",
    )
    args = parser.parse_args()
    cli_start = time.perf_counter()

    cipher_blocks_path = Path(args.cipher_blocks)
    output_path = Path(args.output)

    if args.use_top20_trigrams:
        target_plain_trigrams = DEFAULT_TOP_TRIGRAMS
    else:
        target_plain_trigrams = parse_plain_trigrams_arg(args.plain_trigrams)

    written = run_trigram_attack(
        cipher_blocks_path=str(cipher_blocks_path),
        output_path=str(output_path),
        min_score=args.min_score,
        limit=args.limit,
        cipher_prefix_len=args.cipher_prefix_len,
        best_output_path=args.best_output,
        target_plain_trigrams=target_plain_trigrams,
        count_only=args.count_only,
        benchmark_c_permutations=args.benchmark_c_permutations,
        best_only=args.best_only,
        progress_file=args.progress_file,
        progress_every_c=args.progress_every_c,
        best_event_file=args.best_event_file,
        cipher_text_file=args.cipher_text_file,
    )
    cli_elapsed = time.perf_counter() - cli_start
    if not args.count_only:
        print(f"Done. Wrote {written} candidates to: {output_path}")
    print(f"Total script runtime: {cli_elapsed:.2f}s ({format_duration(cli_elapsed)})")
