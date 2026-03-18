import itertools
import numpy as np
from hillcipher import HillCipher
from collections import Counter
from math import gcd

# Najpogostejši trigrama v angleščini s procenti
COMMON_TRIGRAMS_WITH_FREQ = [
    ("THE", 0.0267),  # 1.81% - 3.67%
    ("AND", 0.0122),  # 0.73% - 1.70%
    ("ING", 0.0089),  # 0.72% - 1.06%
    ("ENT", 0.0042),  # 0.42%
    ("ION", 0.0042),  # 0.42%
    ("HER", 0.0036),  # 0.36%
    ("FOR", 0.0034),  # 0.34%
    ("THA", 0.0033),  # 0.33%
    ("INT", 0.0032),  # 0.32%
    ("ERE", 0.0031),  # 0.31%
    ("TIO", 0.0031),  # 0.31%
    ("TER", 0.0030),  # 0.30%
    ("EST", 0.0028),  # 0.28%
    ("ERS", 0.0028),  # 0.28%
    ("ATI", 0.0026),  # 0.26%
    ("HAT", 0.0026),  # 0.26%
    ("ATE", 0.0025),  # 0.25%
    ("ALL", 0.0025),  # 0.25%
    ("ETH", 0.0024),  # 0.24%
    ("HES", 0.0024),  # 0.24%
]

# Samo trigrama za hitrejše iskanje
COMMON_TRIGRAMS = [t[0] for t in COMMON_TRIGRAMS_WITH_FREQ]
TRIGRAM_FREQUENCIES = {t[0]: t[1] for t in COMMON_TRIGRAMS_WITH_FREQ}

class KeyFinder:
    """
    Iskanje ključa Hillove šifre z analizo najpogostejših trigramov
    """
    
    def __init__(self, ciphertext):
        """
        Args:
            ciphertext: Šifrirano besedilo (samo črke, brez presledkov)
        """
        self.ciphertext = ciphertext.upper()
        self.mod = 26
        
    def text_to_numbers(self, text):
        """Pretvorka besedila v številke"""
        return np.array([ord(char) - ord('A') for char in text], dtype=int)
    
    def numbers_to_text(self, numbers):
        """Pretvorka števk v besedilo"""
        return ''.join(chr((num % 26) + ord('A')) for num in numbers)
    
    def get_trigrams(self, text):
        """
        Ekstraktuj trigrame iz besedila
        
        Returns:
            Counter z najpogostejšimi trigrama
        """
        trigrams = []
        for i in range(len(text) - 2):
            trigrams.append(text[i:i+3])
        return Counter(trigrams)
    
    def _mod_inverse(self, a, m):
        """Izračun modularnega inverza"""
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd_val, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd_val, x, y
        
        _, x, _ = extended_gcd(a % m, m)
        return (x % m + m) % m
    
    def _determinant_invertible(self, matrix, mod):
        """Preverka, ali je determinanta invertibilna"""
        det = int(round(np.linalg.det(matrix)))
        det = det % mod
        return gcd(det, mod) == 1
    

    def solve_key_from_pairs(self, cipher_trigrams_list, plain_trigrams_list):
        """
        Reši ključno matriko iz parov šifriranih in originalnih trigramov
        
        Če imamo 3 pare trigramov, imamo 9 enačb, ki jih lahko rešimo
        za 9 koeficientov matrike 3x3.
        
        C = K * P (mod 26), kjer je:
        C - šifrirani trigram (3x1 vektor)
        K - ključna matrika (3x3)
        P - originalni trigram (3x1 vektor)
        
        Args:
            cipher_trigrams_list: Lista šifriranih trigramov
            plain_trigrams_list: Lista originalnih trigramov
            
        Returns:
            Ključna matrika ali None
        """
        if len(cipher_trigrams_list) < 3 or len(plain_trigrams_list) < 3:
            return None
        
        try:
            # Konvertuj trigrame v vektorje
            C_vecs = [self.text_to_numbers(ct).reshape(-1, 1) for ct in cipher_trigrams_list[:3]]
            P_vecs = [self.text_to_numbers(pt).reshape(-1, 1) for pt in plain_trigrams_list[:3]]
            
            # Sestavi matriko originalnih trigramov (3x3)
            # Vsak stolpec je en trigram
            P_matrix = np.hstack(P_vecs)  # 3x3
            C_matrix = np.hstack(C_vecs)  # 3x3
            
            # Preverka, ali je P_matrix invertibilna
            if not self._determinant_invertible(P_matrix, self.mod):
                return None
            
            # Reši: C = K * P => K = C * P^-1
            P_inv = self._matrix_inverse_mod(P_matrix, self.mod)
            K = (C_matrix @ P_inv) % self.mod
            
            # Preverka, ali je K invertibilna
            if not self._determinant_invertible(K, self.mod):
                return None
            
            return K.astype(int)
            
        except Exception as e:
            return None
    
    def _adjugate_3x3(self, matrix):
        """Izračun adjungirane matrike za 3x3 matriko"""
        m = matrix
        adj = np.array([
            [m[1,1]*m[2,2] - m[1,2]*m[2,1], m[0,2]*m[2,1] - m[0,1]*m[2,2], m[0,1]*m[1,2] - m[0,2]*m[1,1]],
            [m[1,2]*m[2,0] - m[1,0]*m[2,2], m[0,0]*m[2,2] - m[0,2]*m[2,0], m[0,2]*m[1,0] - m[0,0]*m[1,2]],
            [m[1,0]*m[2,1] - m[1,1]*m[2,0], m[0,1]*m[2,0] - m[0,0]*m[2,1], m[0,0]*m[1,1] - m[0,1]*m[1,0]]
        ], dtype=int)
        return adj
    
    def _matrix_inverse_mod(self, matrix, mod):
        """Izračun inverzne matrike modulo"""
        det = int(round(np.linalg.det(matrix)))
        det = det % mod
        
        if gcd(det, mod) != 1:
            raise ValueError(f"Determinanta {det} ni invertibilna modulo {mod}")
        
        det_inv = self._mod_inverse(det, mod)
        adj = self._adjugate_3x3(matrix)
        inv_matrix = (det_inv * adj) % mod
        return inv_matrix
    
    def trigram_frequency_attack(self, cipher_text, top_n=5):
        """
        Napad na osnovu frekvencije trigramov
        
        Primerjaj najpogostejše trigrame iz šifriranog teksta
        sa poznatim trigramima anglešćine
        
        Args:
            cipher_text: Šifrirano besedilo
            top_n: Koiko trigramov da pokušamo
            
        Returns:
            Lista potencialnih ključeva s ocenama
        """
        # Ekstraktuuj trigrame iz šifriranog teksta
        cipher_trigrams = self.get_trigrams(cipher_text)
        most_common_cipher = cipher_trigrams.most_common(top_n * 2)
        
        print("=" * 70)
        print("NAJPOPULARNIJE ŠIFRIRANE TRIGRAME:")
        print("=" * 70)
        for i, (trigram, count) in enumerate(most_common_cipher[:top_n], 1):
            print(f"{i}. {trigram}: {count}x")
        print()
        
        candidates = []
        
        # Pokušaj različite kombinacije mapiranja
        # Pretpostavljamo da je najčešća šifrirana trigram "THE", druga "AND", itd.
        from itertools import combinations, permutations
        
        most_common_plain = COMMON_TRIGRAMS[:top_n]
        cipher_top = [t[0] for t in most_common_cipher[:top_n]]
        
        # Za svaki mogući permutaciju top_n trigramov
        print("Pokušavam različite mapiranja...")
        print("(Ovo može potrajati...)")
        print()
        
        count = 0
        for perm in itertools.permutations(most_common_plain, top_n):
            count += 1
            if count % 100 == 0:
                print(f"  Pokušaj #{count}...", end='\r')
            
            # Reši ključ sa ovim mapiranjem
            key = self.solve_key_from_pairs(cipher_top, list(perm))
            
            if key is not None:
                try:
                    # Pokušaj dekodirati celo besedilo
                    cipher_obj = HillCipher(key)
                    plaintext = cipher_obj.decrypt(cipher_text)
                    
                    # Oceni plaintext
                    score = self._score_plaintext(plaintext)
                    
                    if score > 0.1:
                        candidates.append({
                            'key': key.tolist(),
                            'plaintext': plaintext,
                            'score': score,
                            'mapping': dict(zip(cipher_top, perm))
                        })
                        
                        print(f"\n✓ Pronađen kandidat sa ocenom {score:.4f}")
                        print(f"  Mapiranje: {dict(zip(cipher_top, perm))}")
                        
                except Exception as e:
                    pass
        
        print(f"\nUkupno poskusov: {count}")
        
        # Sortiraj po oceni
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates
    
    def _score_plaintext(self, plaintext):
        """
        Oceni kvaliteto plaintext-a v angleščini
        
        Preveri:
        1. Pogostost znanih besed
        2. Pogostost znanih trigramov
        3. Pogostost samoglasnikov
        
        Returns:
            Točka od 0 do 1
        """
        # Znane besede v angleščini
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
                       'WAS', 'ONE', 'OUR', 'OUT', 'HAD', 'HAS', 'HIS', 'HOW', 'ITS', 'MAY',
                       'BE', 'NO', 'DO', 'GO', 'IF', 'IN', 'IS', 'IT', 'OF', 'ON', 'OR', 'TO',
                       'UP', 'AS', 'BY', 'AT', 'THAT', 'WITH', 'THIS', 'HAVE', 'FROM', 'THEY'}
        
        score = 0.0
        
        # Preveri pogostost znanih besed
        for i in range(len(plaintext) - 1):
            for length in [2, 3, 4, 5]:
                if i + length <= len(plaintext):
                    word = plaintext[i:i+length]
                    if word in common_words:
                        words_in_text.add(word)
                        score += 0.05
        
        # Preveri trigrame
        tri = self.get_trigrams(plaintext)
        for trigram, count in tri.most_common(10):
            if trigram in COMMON_TRIGRAMS:
                score += 0.02 * count
        
        # Preveri samoglasnike (bi morali biti prisotni)
        vowels = "AEIOU"
        vowel_count = sum(1 for char in plaintext if char in vowels)
        vowel_ratio = vowel_count / len(plaintext) if plaintext else 0
        
        if 0.3 < vowel_ratio < 0.5:  # Razumna razmerka
            score += 0.2
        
        # Normalizacija
        return min(score / 10, 1.0)


if __name__ == "__main__":
    print("Iskanje ključa Hillove šifre - Trigram analiza")
    print("=" * 70)
    
    # Šifrirano sporočilo
    ciphertext = "yfyvrbjcpiccrpgiahhegyfyqkprgidcawgqeggpnwmyytyb" + \
                 "htobbucbhnuwvhqiibccrlexvamcclcacvfomapwqintrtdm" + \
                 "plxummjiueczfhgbpsbwkdpxrbkialqmihcpntpjcgxgmrfe" + \
                 "puqwpgrrmywuhsevjvpvrtbmyfyerqzhaftkttxtyifxdzkv" + \
                 "dpnjsaneheaakyxkoawwfnnljicpzuuojialqmifokklggdc" + \
                 "sfebpqkocbmmccanvawmhvnpccrjkghfjlanxwijjdvfgbss" + \
                 "ygqhxzzkvuuizzlkogncksbkchqbbuntroovbhylwwtsfpeo" + \
                 "sivhafjjbfgubdubpdjzqwahwkpjaotnactupnnlmllcifxd" + \
                 "oupcrkpttfoktpxthqtgiyrgunuffesokzsoyfyhnxbmzfgu" + \
                 "evjxwijjdvfgbssygqhxzzkvuuizzlhqazacpivnwglspcgr" + \
                 "ybesfrferqsljtzwcdcyxrdpevjewckdfdpnjukwnwdvrjxt" + \
                 "ytqfzacmknppprxbtqsspeqpvnlwojxnjtxluojhsggfsohi" + \
                 "raufpwpfpeuxbjfcaazbijawikiqxbnkutzinancszaobfsr" + \
                 "nanpcwrhsjbgjicgfqbbubpyyvndkygqksoyzxkorlsilwqi" + \
                 "ntrahsdjqbssricxxhcmljhrdmitpruilzzuvwgmqregackn" + \
                 "fhgbpsbwkxnkvzvyfygpojajzorfccdlqorlxwijjdvfgbss" + \
                 "ygqhxzzkvuuizzlbasgnkaonjzqgoigdcsfebfsjahjpfcid" + \
                 "eahpdbrwucivyhinaauzhjavpcrzfbppqrwuyfyhfabuibbu" + \
                 "dltyfycivyzhesoekiuyxtsqrdfbssbrefakbvyfaispcvay" + \
                 "cemlspfxicivypwhnocjwtmkhcpbasybvbfficxvsugijxzw" + \
                 "cahqofxigpfpznpyfyjictyfhuotyfloqrvwncuyfyvrbjcp" + \
                 "iccrpgiahhegyfyqkprgibxxjiusluqyzgebtrobvyumfbou" + \
                 "gebpmaaeqpftsbygnkwgvlgqvaycemliesbkyrltgibeluui" + \
                 "zzlasqylpbfficxnehpomcszpfpbarlureotnkyyfykzkgij" + \
                 "tqqyeadnepstfxicivybrreqbjmfvhectxbubbuqtuyfyalm" + \
                 "uqzpcrjahjrchauzbxtsfdcloenlnpfdznehgwbpbryfybln" + \
                 "nkynocwbbswjayzdtianaohcbttliedtibpcbiujkqfbkvfo" + \
                 "hasyfyrplugiwqbtihfcchquazespcumfdpxbbhlzkwqqwsa" + \
                 "nbeinzhinwurtsqrhxnkiwqd"
    
    print(f"Šifrirano sporočilo ({len(ciphertext)} črk):")
    print(ciphertext[:80] + "...")
    print()
    
    finder = KeyFinder(ciphertext)
    
    # Trigram analiza
    trigrams = finder.get_trigrams(ciphertext)
    print("Najpogostejši trigrama v šifriranem besedilu:")
    for trigram, count in trigrams.most_common(10):
        print(f"  {trigram}: {count}x")
    print()
    
    # Trigram frequency attack
    print("Začetek trigram frequency attack...")
    print()
    candidates = finder.trigram_frequency_attack(ciphertext, top_n=5)
    
    print()
    print("=" * 70)
    print("KANDIDATI ZA KLJUČ (sortirano po oceni):")
    print("=" * 70)
    
    if candidates:
        for i, candidate in enumerate(candidates[:5], 1):
            print(f"\n{'='*70}")
            print(f"KANDIDAT #{i} | Ocena: {candidate['score']:.4f}")
            print(f"{'='*70}")
            print(f"\nKlučna matrika:")
            for row in candidate['key']:
                print(f"  {row}")
            print(f"\nMapiranje trigramov:")
            for cipher_t, plain_t in candidate['mapping'].items():
                print(f"  {cipher_t} -> {plain_t}")
            print(f"\nDekodovano sporočilo (prvih 150 znakov):")
            print(f"  {candidate['plaintext'][:150]}")
    else:
        print("Nobeden kandidat ni najden. Poskusi z drugačnimi parametri.")
