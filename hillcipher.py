import numpy as np
from math import gcd
from functools import reduce

class HillCipher:
    """
    Hillova šifra - implementacija za kodiranje in dekodiranje sporočil
    """
    
    def __init__(self, key_matrix):
        """
        Inicijalizacija s ključno matrico
        
        Args:
            key_matrix: numpy array matrike 3x3
        """
        self.key_matrix = np.array(key_matrix, dtype=int)
        self.n = self.key_matrix.shape[0]  # Velikost matrike (3)
        self.mod = 26  # Z26 - abeceda
        
    def _mod_inverse(self, a, m):
        """
        Izračun modularnega inverza a^-1 mod m
        """
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd_val, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd_val, x, y
        
        _, x, _ = extended_gcd(a % m, m)
        return (x % m + m) % m
    
    def _matrix_inverse_mod(self, matrix, mod):
        """
        Izračun inverzne matrike modulo
        """
        # Determinanta matrike
        if self.n == 3:
            det = int(round(np.linalg.det(matrix)))
        else:
            det = int(round(np.linalg.det(matrix)))
        
        det = det % mod
        
        # Preverka, ali je determinanta invertibilna
        if gcd(det, mod) != 1:
            raise ValueError(f"Determinanta {det} ni invertibilna modulo {mod}")
        
        det_inv = self._mod_inverse(det, mod)
        
        # Izračun adjungirane matrike
        if self.n == 3:
            adj = self._adjugate_3x3(matrix)
        else:
            adj = self._adjugate_nxn(matrix)
        
        # Inverzna matrika = det_inv * adj mod 26
        inv_matrix = (det_inv * adj) % mod
        return inv_matrix
    
    def _adjugate_3x3(self, matrix):
        """
        Izračun adjungirane matrike za 3x3 matriko
        """
        m = matrix
        adj = np.array([
            [m[1,1]*m[2,2] - m[1,2]*m[2,1], m[0,2]*m[2,1] - m[0,1]*m[2,2], m[0,1]*m[1,2] - m[0,2]*m[1,1]],
            [m[1,2]*m[2,0] - m[1,0]*m[2,2], m[0,0]*m[2,2] - m[0,2]*m[2,0], m[0,2]*m[1,0] - m[0,0]*m[1,2]],
            [m[1,0]*m[2,1] - m[1,1]*m[2,0], m[0,1]*m[2,0] - m[0,0]*m[2,1], m[0,0]*m[1,1] - m[0,1]*m[1,0]]
        ], dtype=int)
        return adj
    
    def _adjugate_nxn(self, matrix):
        """
        Izračun adjungirane matrike za poljubno matriko
        """
        n = matrix.shape[0]
        adj = np.zeros_like(matrix, dtype=int)
        for i in range(n):
            for j in range(n):
                # Izračun minorja
                minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
                det_minor = int(round(np.linalg.det(minor)))
                adj[j, i] = ((-1) ** (i + j)) * det_minor
        return adj
    
    def text_to_numbers(self, text):
        """
        Pretvorba besedila v številke (A=0, B=1, ..., Z=25)
        """
        return [ord(char.upper()) - ord('A') for char in text]
    
    def numbers_to_text(self, numbers):
        """
        Pretvorka števk nazaj v besedilo
        """
        return ''.join(chr((num % 26) + ord('A')) for num in numbers)
    
    def decrypt(self, ciphertext):
        """
        Dekodiranje šifriranega besedila
        
        Args:
            ciphertext: Šifrirano besedilo (samo črke, brez presledkov)
            
        Returns:
            Originalno besedilo
        """
        # Izračun inverzne matrike
        key_inv = self._matrix_inverse_mod(self.key_matrix, self.mod)
        
        # Pretvorka šifriranega besedila v številke
        cipher_numbers = self.text_to_numbers(ciphertext)
        
        # Dekodiranje po blokah
        plaintext_numbers = []
        for i in range(0, len(cipher_numbers), self.n):
            block = np.array(cipher_numbers[i:i+self.n], dtype=int)
            decrypted_block = (key_inv @ block) % self.mod
            plaintext_numbers.extend(decrypted_block)
        
        # Pretvorka nazaj v besedilo
        plaintext = self.numbers_to_text(plaintext_numbers)
        return plaintext
    
    def encrypt(self, plaintext):
        """
        Kodiranje besedila
        
        Args:
            plaintext: Originalno besedilo (samo črke, brez presledkov)
            
        Returns:
            Šifrirano besedilo
        """
        # Pretvorka besedila v številke
        plain_numbers = self.text_to_numbers(plaintext)
        
        # Kodiranje po blokah
        ciphertext_numbers = []
        for i in range(0, len(plain_numbers), self.n):
            block = np.array(plain_numbers[i:i+self.n], dtype=int)
            encrypted_block = (self.key_matrix @ block) % self.mod
            ciphertext_numbers.extend(encrypted_block)
        
        # Pretvorka nazaj v besedilo
        ciphertext = self.numbers_to_text(ciphertext_numbers)
        return ciphertext


if __name__ == "__main__":
    # Primer uporabe
    print("Hillova šifra - Dekodiranje")
    print("=" * 50)
    
    # Primer ključne matrike (3x3)
    key = [
        [1, 2, 3],
        [0, 1, 4],
        [5, 1, 2]
    ]
    
    cipher = HillCipher(key)
    
    # Primer: dekodiranje
    # ciphertext = "EXAMPLE"
    # plaintext = cipher.decrypt(ciphertext)
    # print(f"Šifrirano besedilo: {ciphertext}")
    # print(f"Dekodovano besedilo: {plaintext}")
