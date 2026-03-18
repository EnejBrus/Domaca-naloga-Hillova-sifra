#!/usr/bin/env python3
"""
Skripta za čuvanje i pregled svih pronađenih ključeva za Hill cipher
"""

import json
from keyfinder import KeyFinder

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

finder = KeyFinder(ciphertext)

# Išči ključeve
candidates = finder.trigram_frequency_attack(ciphertext, top_n=7)

# Spremi rezultate u JSON datoteku
results = []
for i, candidate in enumerate(candidates):
    results.append({
        'rank': i + 1,
        'score': candidate['score'],
        'key': candidate['key'],
        'plaintext': candidate['plaintext'][:300],  # Prvo 300 znakov
        'mapping': candidate['mapping']
    })

with open('keyfinder_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Pronađeno {len(candidates)} mogućih ključeva")
print("Rezultati su sačuvani u 'keyfinder_results.json'")
print(f"\nTop 5 kandidata:")
for i, candidate in enumerate(candidates[:5], 1):
    print(f"\n{i}. Ocena: {candidate['score']:.6f}")
    print(f"   Matrika: {candidate['key']}")
    print(f"   Tekst: {candidate['plaintext'][:100]}...")
