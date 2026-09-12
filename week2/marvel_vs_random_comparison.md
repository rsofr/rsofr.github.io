# Comparaison Structurelle : Réseau Marvel Réel vs Réseau Aléatoire

| Modèle         | N   | M    | ⟨k⟩   | GCC (nœuds) | GCC (%) | Distance GCC | Isolats | k_max        | C           |
|:---------------|----:|-----:|------:|------------:|--------:|-------------:|--------:|-------------:|------------:|
| Real Marvel    | 303 | 1434 |  9.47 |         277 |  91.42% |         2.67 |      17 |          106 |        0.31 |
| Random G(n, m) | 303 | 1784 | 11.78 |         303 | 100.00% |  2.58 ± 0.01 |       0 | 22.16 ± 1.35 | 0.04 ± 0.01 |


### Observations structurelles clés :

1. **Absence de Hubs dans le réseau aléatoire** : Dans Random G(n, m), le degré maximal moyen est de 22.16 ± 1.35 (distribution poissonnienne bornée), tandis que Real Marvel culmine à k_max = 106 (hubs majeurs reliant plus du tiers du réseau).
2. **Clustering très élevé dans le monde réel** : Le clustering de Marvel (0.31) est près de 8 fois supérieur à celui du réseau aléatoire (0.04 ± 0.01), démontrant une forte fermeture triadique et organisation en communautés.
3. **Présence d'isolats réels** : Real Marvel compte 17 personnages isolés (5.61%), alors que Random G(n, m) à ⟨k⟩ = 11.78 intègre la totalité des nœuds dans sa composante géante (0 isolat).
4. **Effet Petit Monde (Small World)** : La distance moyenne reste très faible dans les deux cas (2.67 vs 2.58), confirmant que les deux topologies sont du type 'petit monde'.
