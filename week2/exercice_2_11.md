# Exercice 2.11 - Go nuts with your LLM : Le Paradoxe de l'Amitié et les Modèles Nuls dans le Multivers Marvel

**Auteur :** Romeo Sofia  
**Date :** 12 Septembre 2026  
**Cours :** DTU 02805 – Social Graphs and Interactions  
**Temps de lecture :** ~7 min  

---

## Introduction : Quand la Topologie Bouscule l'Intuition

Vous êtes-vous déjà senti moins populaire que vos amis sur les réseaux sociaux ? Rassurez-vous : **87,4 % des super-héros Marvel ressentent exactement la même chose**. Dans les pages de comics comme sur les réseaux sociaux réels, vos amis ont presque systématiquement plus d'amis que vous.

Dans le cadre de cet **Exercice 2.11 ("Go nuts with your LLM")**, nous explorons la **composante géante connexe (GCC)** du réseau Marvel Comics ($N = 277$ héros, $M = 1421$ arêtes non-orientées) afin de disséquer deux questions fondamentales de la théorie des réseaux :

1. **Le paradoxe de l'amitié chez les super-héros :** Pourquoi vos alliés sont-ils mathématiquement plus connectés que vous ? Quels hubs causent ce paradoxe et qui parvient à y échapper ?
2. **Le Shuffle Test sur le clustering (Modèle Nul) :** Le regroupement des héros en factions (Avengers, X-Men) est-il un simple sous-produit de l'existence de quelques stars ultra-connectées, ou le reflet indéniable d'une véritable structure communautaire ?

---

## 1. Le Paradoxe de l'Amitié chez les Super-Héros

Formulé par le sociologue Scott L. Feld en 1991, le paradoxe de l'amitié énonce que *« vos amis ont en moyenne plus d'amis que vous »*.

### A. Les résultats empiriques sur Marvel

- **Degré moyen d'un super-héros choisi au hasard ($\langle k \rangle$) :**  
  $$\langle k \rangle = 10{,}26 \text{ connexions}$$
- **Degré moyen d'un voisin choisi au hasard ($\langle k_{\text{voisin}} \rangle$) :**  
  $$\langle k_{\text{voisin}} \rangle = \frac{\langle k^2 \rangle}{\langle k \rangle} = 22{,}25 \text{ connexions}$$
- **Moyenne des degrés moyens des voisins par nœud ($\langle k_{\text{nn}} \rangle$) :**  
  $$\langle k_{\text{nn}} \rangle = 24{,}90 \text{ connexions}$$
- **Ratio d'amplification :** Les partenaires d'un super-héros ont en moyenne **plus de deux fois plus d'alliés ($\times 2{,}17$)** que lui-même !

### B. Pourquoi ce paradoxe survient-il ? (Démonstration mathématique)

Il ne s'agit pas d'un biais cognitif, mais d'un biais d'échantillonnage géométrique inévitable. Lorsque l'on suit une arête aléatoire dans un réseau, la probabilité d'atterrir sur un nœud de degré $k$ est proportionnelle à son propre degré $k$.

L'espérance du degré d'un voisin s'écrit formellement :
$$\langle k_{\text{voisin}} \rangle = \frac{\langle k^2 \rangle}{\langle k \rangle} = \langle k \rangle + \frac{\sigma_k^2}{\langle k \rangle}$$

Dans le réseau Marvel, la variance des degrés est très forte ($\sigma_k^2 = 123{,}04$). Puisque $\sigma_k^2 > 0$, $\langle k_{\text{voisin}} \rangle$ est **strictement supérieur** à $\langle k \rangle$. Plus le réseau contient des super-hubs hétérogènes, plus le paradoxe est violent.

### C. Qui sont les Hubs responsables du paradoxe ?

Les "coupables" sont les super-héros omnidirectionnels qui figurent dans le voisinage d'un très grand nombre d'autres personnages :

| Rang | Super-Héros (Hub) | Degré Réel ($k$) | Degré Moyen des Voisins ($k_{\text{nn}}$) | Excédent ($k - k_{\text{nn}}$) |
|:----:|:-------------------|:----------------:|:-----------------------------------------:|:------------------------------:|
| 1 | **Spider-Man** | **106** | 14.92 | **+91.08** |
| 2 | **Hulk** | **65** | 15.28 | **+49.72** |
| 3 | **Wolverine** | **63** | 17.84 | **+45.16** |
| 4 | **Doctor Strange** | **57** | 18.98 | **+38.02** |
| 5 | **Deadpool** | **41** | 19.71 | **+21.29** |
| 6 | **She-Hulk** | **37** | 19.76 | **+17.24** |
| 7 | **Black Panther** | **31** | 17.03 | **+13.97** |

### D. Qui échappe au paradoxe ?

Un super-héros ne subit pas le paradoxe si $k_i \ge k_{\text{nn}, i}$.
- **Seulement 35 super-héros sur 277 (12,6 %)** sont immunisés contre le paradoxe. Ce sont précisément les méga-hubs listés ci-dessus.
- **242 super-héros (87,4 %)** subissent le paradoxe de plein fouet.

---

> ### 🖼️ [INSÉRER ICI LE GRAPHIQUE 1 : `friendship_paradox.png`]
> **Légende :** *(A) Dispersion nœud par nœud montrant les 87,4 % de victimes dans la zone rouge au-dessus de la droite $y = x$, et les 12,6 % d'hubs immunisés sous la droite. (B) Diagramme en barres comparant le degré moyen d'un héros (10,26) à celui d'un voisin (22,25).*

---

## 2. Le Shuffle Test : Le Hasard Structurel Explique-t-il les Regroupements ?

Le réseau Marvel réel affiche un coefficient de clustering moyen remarquable :
$$\langle C_{\text{réel}} \rangle = 0{,}3199 \quad (\text{Transitivité } T = 0{,}1823)$$

Mais cette cohésion est-elle simplement due au fait que des personnages très connectés forment mécaniquement des triangles fortuits ?

### A. Protocole du Modèle Nul préservant les degrés

Pour tester cette hypothèse, nous utilisons un **modèle de configuration par brassage d'arêtes** (`nx.double_edge_swap`) :
- On permute des paires d'arêtes indépendantes $14\,210$ fois par réseau ($10 \times M$).
- **La distribution exacte des degrés de chaque personnage est préservée à 100 %**.
- On répète la simulation pour générer **100 modèles nuls indépendants**.

### B. Résultats du test d'hypothèse

- **Moyenne du clustering nul :** $\langle C_{\text{nul}} \rangle = 0{,}1554 \pm 0{,}0084$ (min : $0{,}1399$, max : $0{,}1816$)
- **Clustering réel :** $\langle C_{\text{réel}} \rangle = 0{,}3199$
- **Z-score :**
  $$z = \frac{0{,}3199 - 0{,}1554}{0{,}0084} = \mathbf{+19{,}57 \ \sigma}$$
- **$p$-value empirique :** $p = 0{,}0000 \quad (p < 0{,}001)$

---

> ### 🖼️ [INSÉRER ICI LE GRAPHIQUE 2 : `null_model_clustering_shuffle.png`]
> **Légende :** *Distribution nulle du coefficient de clustering moyen sur 100 modèles brassés (courbe bleue) comparée à la valeur réelle de l'univers Marvel Comics (ligne rouge verticale à $C = 0{,}3199$). L'écart gigantesque de $+19{,}57$ écarts-types rejette totalement le hasard.*

---

### C. Interprétation : Pourquoi le hasard ne suffit pas ?

Avoir des super-hubs permet d'augmenter le clustering de $0{,}04$ (modèle d'Erdős-Rényi sans hubs) à environ $0{,}155$. Cependant, le clustering réel culmine à **$0{,}320$** !

Ce résultat démontre de façon statistiquement irréfutable l'existence d'une **fermeture triadique dense et d'une structure en communautés réelles** :
- Les super-héros ne se connectent pas au hasard des probabilités de degré.
- Ils appartiennent à des **équipes éditoriales pérennes** (Avengers, X-Men, 4 Fantastiques, Defenders) qui génèrent des cliques et des triangles denses bien au-delà de ce que la seule séquence de degrés pourrait produire.

---

## Conclusion & Enseignements

1. **Le paradoxe de l'amitié n'épargne personne :** Plus de $87 \%$ des personnages de comics ont des alliés plus populaires qu'eux, conséquence inévitable de la forte hétérogénéité des degrés.
2. **Le multivers Marvel est hautement communautaire :** Le test du modèle nul avec un Z-score de $+19{,}57$ confirme que la narration de Marvel est structurée autour de collectifs soudés et non d'interactions aléatoires.

*Article rédigé par Romeo Sofia – DTU 02805 Social Graphs & Interactions.*
