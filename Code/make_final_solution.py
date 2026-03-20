from pathlib import Path
import re
import numpy as np

from Code.yfy_fixed_k_search_timed import (
    parse_raw_ciphertext,
    matrix_inverse_mod_3x3,
    decrypt_with_key,
    format_matrix,
)


def main() -> None:
    k = np.array([[7, 3, 0], [5, 0, 10], [4, 16, 11]], dtype=int)
    k_inv = matrix_inverse_mod_3x3(k, 26)

    cipher_letters, _ = parse_raw_ciphertext(Path("cipher_imput_raw.txt"), 456)
    plaintext = decrypt_with_key(cipher_letters, k)

    report_text = Path("yfy_fixed_k_candidates_scored_timed.txt").read_text(encoding="utf-8")
    match = re.search(r"Runtime seconds:\s*([0-9]+(?:\.[0-9]+)?)", report_text)
    runtime = match.group(1) if match else "N/A"

    lines = [
        "FINAL SOLUTION",
        "=" * 80,
        f"K matrix: {format_matrix(k)}",
        f"K inverse mod 26: {format_matrix(k_inv)}",
        f"Runtime for matrix search (seconds): {runtime}",
        "",
        "Deciphered text:",
        plaintext,
    ]

    Path("final_solution.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote final_solution.txt")


if __name__ == "__main__":
    main()
