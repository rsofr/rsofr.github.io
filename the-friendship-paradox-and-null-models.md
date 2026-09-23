# The Friendship Paradox and Null Models in the Marvel Multiverse

**Author:** Romeo Sofia  
**Date:** September 12, 2026  
**Course:** DTU 02805 – Social Graphs and Interactions  
**Reading Time:** ~7 min  

---

## Introduction: When Topology Defies Intuition

Have you ever felt less popular than your friends on social media? Take comfort: **87.4% of Marvel superheroes feel the exact same way**. In comic book pages just as on real-world social networks, your friends almost systematically have more friends than you do.

In this open-ended assignment ("Go nuts with your LLM"), we explore the **Giant Connected Component (GCC)** of the Marvel Comics network ($N = 277$ heroes, $M = 1,421$ undirected edges) to dissect two fundamental questions in network theory:

1. **The Superhero Friendship Paradox:** Why are your allies mathematically more connected than you are? Which hubs drive this paradox, and who manages to escape it?
2. **The Clustering Shuffle Test (Null Model):** Is the clustering of heroes into distinct factions (Avengers, X-Men) merely an accidental byproduct of a few ultra-connected superstar hubs, or does it reflect genuine, non-random community structure?

---

## 1. The Friendship Paradox in Superheroes

First formulated by sociologist Scott L. Feld in 1991, the friendship paradox states that *"on average, your friends have more friends than you do."*

### A. Empirical Findings in the Marvel Network

- **Average degree of a randomly chosen superhero ($\langle k \rangle$):**  
  $$\langle k \rangle = 10.26 \text{ connections}$$
- **Average degree of a randomly chosen neighbor ($\langle k_{\text{neighbor}} \rangle$):**  
  $$\langle k_{\text{neighbor}} \rangle = \frac{\langle k^2 \rangle}{\langle k \rangle} = 22.25 \text{ connections}$$
- **Mean of the average neighbor degrees per node ($\langle k_{\text{nn}} \rangle$):**  
  $$\langle k_{\text{nn}} \rangle = 24.90 \text{ connections}$$
- **Amplification ratio:** A superhero's partners possess, on average, **more than twice as many allies ($\times 2.17$)** as the hero themselves!

### B. Why Does This Paradox Arise? (Mathematical Derivation)

This is not a cognitive bias, but an inevitable geometric sampling bias. When traversing a random edge in a network, the probability of arriving at a node of degree $k$ is strictly proportional to its own degree $k$.

The expected degree of a neighbor is formally expressed as:
$$\langle k_{\text{neighbor}} \rangle = \frac{\langle k^2 \rangle}{\langle k \rangle} = \langle k \rangle + \frac{\sigma_k^2}{\langle k \rangle}$$

In the Marvel network, degree variance is exceptionally high ($\sigma_k^2 = 123.04$). Because $\sigma_k^2 > 0$, $\langle k_{\text{neighbor}} \rangle$ is **strictly greater** than $\langle k \rangle$. The more a network features heavy-tailed degree heterogeneity and mega-hubs, the more pronounced the paradox becomes.

### C. The Hubs Responsible for the Paradox

The "culprits" are the omnidirectional hubs that populate the immediate neighborhood of an overwhelming number of peripheral characters:

| Rank | Superhero (Hub) | Degree ($k$) | Average Neighbor Degree ($k_{\text{nn}}$) | Degree Surplus ($k - k_{\text{nn}}$) |
|:----:|:----------------|:------------:|:-----------------------------------------:|:------------------------------------:|
| 1 | **Spider-Man** | **106** | 14.92 | **+91.08** |
| 2 | **Hulk** | **65** | 15.28 | **+49.72** |
| 3 | **Wolverine** | **63** | 17.84 | **+45.16** |
| 4 | **Doctor Strange** | **57** | 18.98 | **+38.02** |
| 5 | **Deadpool** | **41** | 19.71 | **+21.29** |
| 6 | **She-Hulk** | **37** | 19.76 | **+17.24** |
| 7 | **Black Panther** | **31** | 17.03 | **+13.97** |

### D. Who Escapes the Paradox?

A superhero is immune to the paradox if and only if $k_i \ge k_{\text{nn}, i}$.
- **Only 35 out of 277 superheroes (12.6%)** are immune to the paradox. These are precisely the mega-hubs listed above.
- **242 superheroes (87.4%)** experience the paradox full force.

---

> ### 🖼️ [FIGURE 1: `friendship_paradox.png`]
> **Caption:** *(A) Node-by-node scatter plot showing the 87.4% of victims in the red shaded zone above the diagonal line $y = x$, and the 12.6% of immune hubs below the line. (B) Bar chart comparing the average degree of a hero (10.26) against that of a neighbor (22.25).*

---

## 2. The Shuffle Test: Does Structural Randomness Explain Clustering?

The empirical Marvel network exhibits a remarkably high average clustering coefficient:
$$\langle C_{\text{real}} \rangle = 0.3199 \quad (\text{Transitivity } T = 0.1823)$$

However, is this cohesion simply an artifact of degree heterogeneity—where highly connected characters mechanically form random triangles?

### A. Degree-Preserving Null Model Protocol

To test this hypothesis, we construct a **configuration null model via edge swapping** (`nx.double_edge_swap`):
- Pairs of independent edges are swapped $14,210$ times per realization ($10 \times M$).
- **The exact degree distribution of every single character is preserved at 100%**.
- The simulation is repeated across **100 independent null network realizations**.

### B. Hypothesis Testing Results

- **Null clustering mean:** $\langle C_{\text{null}} \rangle = 0.1554 \pm 0.0084$ (min: $0.1399$, max: $0.1816$)
- **Empirical clustering:** $\langle C_{\text{real}} \rangle = 0.3199$
- **Z-score:**
  $$z = \frac{0.3199 - 0.1554}{0.0084} = \mathbf{+19.57 \ \sigma}$$
- **Empirical $p$-value:** $p = 0.0000 \quad (p < 0.001)$

---

> ### 🖼️ [FIGURE 2: `null_model_clustering_shuffle.png`]
> **Caption:** *Null distribution of the average clustering coefficient over 100 degree-preserving swapped networks (blue density curve) compared to the empirical value of the Marvel Comics universe (solid red vertical line at $C = 0.3199$). The massive deviation of $+19.57$ standard deviations decisively rejects the random configuration hypothesis.*

---

### C. Interpretation: Why Randomness Falls Short

Introducing degree heterogeneity boosts the clustering coefficient from $C \approx 0.04$ (in a purely random Erdős–Rényi graph) to approximately $0.155$. Yet the real Marvel universe reaches **$0.320$**!

This discrepancy provides undeniable statistical proof of **dense triadic closure and genuine community organization**:
- Superheroes do not interact merely based on degree-proportional attachment.
- They belong to **long-standing narrative alliances and editorial teams** (Avengers, X-Men, Fantastic Four, Defenders) that generate tight cliques and triangles far beyond what degree sequences alone can explain.

---

## Conclusion & Key Takeaways

1. **The friendship paradox spares almost no one:** Over $87\%$ of comic book characters have allies who are more popular than themselves, an inevitable mathematical consequence of heavy-tailed degree distributions.
2. **The Marvel Multiverse is intrinsically modular:** The null model randomization test ($z = +19.57\sigma$) confirms that comic book storytelling is organized around tightly knit, collaborative teams rather than decentralized, random encounters.

*Article by Romeo Sofia – DTU 02805 Social Graphs & Interactions.*
