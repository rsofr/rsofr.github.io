# Louvain vs. Infomap on the Philosophers: How Mathematics Shapes the History of Thought

**Author:** Romeo Sofia  
**Date:** September 23, 2026  
**Course:** DTU 02805 – Social Graphs and Interactions  
**Reading Time:** ~9 min  

---

## Introduction: When Algorithms Write Intellectual History

What is a "school of thought"? 

To an intellectual historian, a school is defined by shared questions, fierce debates, mentor-disciple lineages, and common vocabularies. But to a network scientist, a school is an algorithmic partition—a mathematical boundary drawn through thousands of cross-references and citations. 

Yet mathematics is not neutral. How we choose to define a "community" formally dictates how we reconstruct the history of human thought. 

Consider two foundational paradigms in community detection:
1. **The Static Structural Cut (Louvain / Modularity Maximization):** A community is a cluster of thinkers whose mutual connections are denser than expected under a random degree-preserving null model.
2. **The Dynamic Information Flow (Infomap / The Map Equation):** A community is a territory where an intellectual random walker—representing the transmission of ideas, citations, or arguments—spends extended time circulating before escaping to another domain.

In this analytical study, we pit the **Louvain algorithm** against **Infomap** on the Giant Connected Component (GCC) of historical philosophers ($N = 1,374$ thinkers, $M = 9,139$ weighted edges, total weight $W = 15,757.0$). While the two algorithms exhibit broad agreement on the macro-continents of history (Normalized Mutual Information $\mathbf{\text{NMI} \approx 0.69}$), they clash across **41.3% of the network**.

Which algorithm tells the truer history of philosophy? And why does modularity force British empiricists and continental rationalists into the same intellectual marriage, while the map equation tears them apart?

---

## 1. Mathematical Foundations: Density vs. Diffusion

To understand where the algorithms diverge, we must look at what objective functions they optimize.

### A. Louvain: Newman-Girvan Modularity Maximization

The Louvain algorithm optimizes Newman-Girvan modularity $Q$, measuring the fraction of edges falling within communities minus the expected fraction in an equivalent configuration null model:

$$Q = \frac{1}{2m} \sum_{i,j} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)$$

where $A_{ij}$ is the edge weight between philosophers $i$ and $j$, $k_i$ is their weighted degree (strength), $2m = \sum_i k_i$ is twice the total edge weight, and $\delta(c_i, c_j) = 1$ if both share community $c$, and $0$ otherwise.

- **Mechanics:** Fast, two-phase greedy optimization. Phase 1 shifts individual nodes into neighboring communities to maximize local $\Delta Q$. Phase 2 aggregates communities into super-nodes and repeats until modularity plateaus.
- **Topological Bias:** Purely static and edge-density driven.

### B. Infomap: The Map Equation and Shannon Coding

Infomap reformulates community detection as an optimal lossy data compression problem. Guided by the principle that *"to understand a system, trace the flow within it,"* Rosvall and Bergstrom (2008) introduced the **Map Equation**:

$$L(\mathsf{M}) = q_{\curvearrowright} H(\mathcal{Q}) + \sum_{i=1}^m p_{\circlearrowright}^i H(\mathcal{P}^i)$$

where:
- $q_{\curvearrowright} = \sum_{i=1}^m q_{i \curvearrowright}$ is the total per-step probability that a random walker exits any module,
- $H(\mathcal{Q}) = -\sum_{i=1}^m \frac{q_{i \curvearrowright}}{q_{\curvearrowright}} \log_2 \left( \frac{q_{i \curvearrowright}}{q_{\curvearrowright}} \right)$ is the entropy of the module transition codebook (the "index codebook"),
- $p_{\circlearrowright}^i = q_{i \curvearrowright} + \sum_{\alpha \in M_i} p_\alpha$ is the total rate of movement within module $i$ (including the exit step),
- $H(\mathcal{P}^i) = -\frac{q_{i \curvearrowright}}{p_{\circlearrowright}^i} \log_2 \left( \frac{q_{i \curvearrowright}}{p_{\circlearrowright}^i} \right) - \sum_{\alpha \in M_i} \frac{p_\alpha}{p_{\circlearrowright}^i} \log_2 \left( \frac{p_\alpha}{p_{\circlearrowright}^i} \right)$ is the entropy of the intra-module codebook for module $i$,
- $p_\alpha = \frac{k_\alpha}{2m}$ is the stationary ergodic visitation probability of node $\alpha$.

- **Mechanics:** Infomap finds the module partition $\mathsf{M}$ that minimizes $L(\mathsf{M})$, the expected number of bits required to describe an infinite random walk over the network.
- **Topological Bias:** Dynamic and flow-based. Communities are natural trapping basins for intellectual discourse.

---

## 2. Empirical Findings on the Philosophers Network

Running both algorithms on the giant component yields stark structural differences:

| Metric | Louvain Algorithm | Infomap (Map Equation) |
|:-------|:-----------------:|:----------------------:|
| **Objective Function** | Modularity $Q$ | Map Equation $L(\mathsf{M})$ |
| **Optimized Value** | $Q = 0.5645$ | $L(\mathsf{M}) = 8.3876 \text{ bits}$ |
| **Number of Communities** | **10** macro-clusters | **19–28** fine-grained modules |
| **Largest Community Size** | 242 nodes (17.6%) | 153 nodes (11.1%) |
| **Smallest Community Size** | 2 nodes | 3 nodes |
| **Normalized Mutual Information (NMI)** | \multicolumn{2}{c|}{$\mathbf{\text{NMI} = 0.6888}$} |
| **Node Partition Concordance** | \multicolumn{2}{c|}{**58.66% Agreement** (806 nodes) \| **41.34% Disagreement** (568 nodes)} |

### The Macro Communities of Louvain:
1. **Early Modern Rationalists & Empiricists ($n = 242$):** Leibniz, Descartes, Locke, Hume, Newton.
2. **German Idealism & 19th Century Continental ($n = 215$):** Kant, Hegel, Nietzsche, Spinoza, Husserl.
3. **Classical Antiquity ($n = 210$):** Aristotle, Plato, Diogenes Laertius, Heraclitus, Cicero.
4. **Early Analytic, Pragmatism & Science ($n = 164$):** Russell, James, Peirce, Bergson, Einstein.
5. **Christian Scholasticism & Reformation ($n = 162$):** Thomas Aquinas, Augustine of Hippo, Erasmus, Luther.
6. **Islamic Golden Age & Medieval Jewish ($n = 135$):** Avicenna, Averroes, Roger Bacon, Maimonides.
7. **Indian & Eastern Traditions ($n = 95$):** Madhvacharya, The Buddha, Adi Shankara, Ramanuja.
8. **Chinese Classical Thought ($n = 75$):** Confucius, Mencius, Shen Buhai, Zhu Xi.
9. **Marxism & Russian Philosophy ($n = 74$):** Karl Marx, Friedrich Engels, Leo Tolstoy, Vladimir Lenin.
10. **Fourth Way Esotericism ($n = 2$):** P. D. Ouspensky, George Gurdjieff.

---

## 3. Dissecting the Disagreement: The Map of Discord

Where does the 41.3% disagreement come from?

To visualize the divergence without arbitrary label ordering, we solved the bipartite matching problem between partitions using the **Kuhn-Munkres (Hungarian) algorithm** on their contingency overlap matrix.

---

> ### 🖼️ Figure 1: The Anatomy of Algorithmic Disagreement
> ![Network of Disagreement: Louvain vs. Infomap](assets/louvain_vs_infomap_disagreement.png)
> *Figure 1: (Left) Force-directed subnetwork layout of top thinkers. Blue nodes denote method concordance; vibrant coral nodes denote methodological discordance, with key disputed giants highlighted. (Top Right) The modularity resolution limit: how Louvain's monolithic clusters shatter into distinct Infomap flow modules. (Bottom Right) Method disagreement rate across historical eras, showing peak friction in the 18th and 19th centuries.*

---

### The Three Structural Archetypes of Discord:

#### 1. The Resolution Limit Shatters Louvain's Mega-Communities
In 2007, Santo Fortunato and Marc Barthélemy proved that modularity maximization has an intrinsic **resolution limit**: it fails to detect well-defined modules below a characteristic scale proportional to the square root of the total network volume:

$$\text{Scale} \approx \sqrt{2m} = \sqrt{2 \times 15,757} \approx 177.5 \text{ edge weights}$$

Because the philosophers network has over 15,000 edge weights, Louvain cannot distinguish two tightly knit schools if their total degree is below ~175—it glues them together if they share even a handful of bridge edges!

The consequences are dramatic:
- **Louvain's Early Modern Monolith (242 nodes)** merges **Gottfried Wilhelm Leibniz, René Descartes, John Locke, David Hume, and Isaac Newton** into one bucket.
- **Infomap shatters this blob into three authentic traditions**:
  1. *Continental Rationalism* (Leibniz, Descartes, Malebranche, Spinoza)
  2. *British Empiricism & Utilitarianism* (Locke, Hume, Berkeley, John Stuart Mill)
  3. *Enlightenment Political Philosophy* (Jean-Jacques Rousseau, Montesquieu, Thomas Jefferson).

#### 2. The Ancient Classical Dilemma
Louvain bundles all classical thinkers into an undifferentiated Antiquity sack (210 nodes). Infomap separates:
- The **Socratic-Platonic Academy** (Plato, Diogenes Laertius, Socrates)
- The **Peripatetic Aristotelian Tradition** (Aristotle, Simplicius of Cilicia, Theophrastus)
- The **Patristic & Neoplatonic Bridge** (Augustine of Hippo, Origen, Plotinus)

#### 3. Intellectual Polymaths and Boundary Thinkers
The table below highlights the most influential thinkers where the two algorithms disagree:

| Philosopher | Weighted Degree | Historical Era | Louvain Allegiance | Infomap Flow Allegiance | Historical Reality |
|:------------|:---------------:|:---------------|:-------------------|:------------------------|:-------------------|
| **Aristotle** | 521.0 | Antiquity (BC) | Classical Greek | Peripatetic / Scholastic Commentaries | Bridge between Athens, Baghdad, and Christian Paris |
| **David Hume** | 168.0 | 18th Century | Modern Rationalism | British Empiricism | Scottish Enlightenment empiricist, radically distinct from Descartes |
| **Edmund Husserl** | 158.0 | 19th Century | German Idealism | Phenomenology & Existentialism | Founded phenomenology; separated from Hegelian metaphysics |
| **Augustine of Hippo** | 158.0 | 1st–10th c. | Medieval Scholasticism | Patristic / Neoplatonism | Bridged late pagan antiquity and early Christian theology |
| **Martin Heidegger** | 152.0 | 19th/20th c. | German Idealism | 20th c. Existentialism | Fundamental ontology, separated from 19th c. Kantianism |
| **Jean-Jacques Rousseau** | 145.0 | 18th Century | Modern Rationalism | Political Contract Theory | Pre-Romantic political rebel, antithetical to Cartesian mechanism |
| **Henri Bergson** | 114.0 | 19th Century | Analytic / Pragmatism | Continental Vitalism | Nobel laureate in process philosophy, forced by Louvain into Russell's orbit |
| **Albert Einstein** | 101.0 | 19th/20th c. | Analytic / Pragmatism | Philosophy of Science & Physics | Physics revolutionary linked to Mach and Poincaré, not formal logic |

---

## 4. Sharp Analytical Discussion: Which Method Tells the Better History?

When assessing whether Louvain or Infomap provides a superior historiographical model, network science principles deliver an unequivocal answer: **Infomap tells a far superior, more authentic history of philosophy.**

### 1. The Dynamic Nature of Intellectual Lineages
The history of philosophy is not a static block of marble; it is a **directed, dynamic diffusion process**. Ideas flow through space and time. A scholar reads Kant, reacts to Hume, and writes a critique that influences Hegel. 

Modularity treats edges as symmetric spring-like tensions. To Louvain, if Hegel and Nietzsche cite Spinoza extensively, Spinoza must be dragged out of the 17th century and placed in 19th-century Germany! 

In contrast, Infomap models the journey of an intellectual random walker. The Map Equation asks: *if an inquirer starts reading texts and following influence links, where do they linger before breaking through to a new school?* 
A random walker spends extensive time trapped within the self-referential terminology of British Empiricism (sense data, impressions, tabula rasa, skepticism) before crossing the conceptual channel into Cartesian rationalism (innate ideas, cogito, monads). Infomap preserves these natural boundaries because transitioning between them incurs a coding penalty in the index codebook.

### 2. The Resolution Limit as Historiographical Distortion
Modularity's global denominator $2m$ imposes an arbitrary historical horizon. In a large network, modularity operates like a low-resolution telescope: it sees that Europe is distinct from China, and that Antiquity is distinct from the Middle Ages, but it cannot resolve the nuanced schisms that define philosophical progress. 

Claiming that Descartes (the champion of mathematical rationalism) and Hume (the radical empiricist skeptic) belong to the same community is an intellectual distortion. Louvain creates this unnatural merger solely because $2m$ is too large to let them separate. Infomap, by compressing localized codebooks, scales independently of global network volume, detecting compact, specialized movements like Stoicism, Neoplatonism, and the Scottish Enlightenment.

### 3. Transitional Eras Reveal Peak Conflict
As demonstrated in our empirical era analysis (Figure 1, Panel C), method disagreement peaks in the **18th century (62.1%)** and **19th century (56.4%)**. Why? 

These were the centuries of the Enlightenment, the Industrial Revolution, and the scientific explosion—periods characterized by unprecedented intellectual cross-pollination. Polymaths like Adam Smith (moral philosopher and political economist) and Galileo Galilei (natural philosopher and experimentalist) sat directly on the fault lines between declining scholastic traditions and emerging empirical sciences. 

Modularity flattens these complex bridge figures into whichever mega-block has slightly higher edge count. Infomap captures the transition: its codebook structure allows bridge thinkers to serve as gateways between cohesive, distinct modules.

---

## 5. Conclusion & Takeaways

Community detection algorithms are not neutral calculators; they are **computational historiographers**. 

- **Louvain is a cartographer of macro-continents:** It provides an excellent bird's-eye view of broad cultural and civilizational eras (Antiquity, Islamic Golden Age, Medieval Christendom, East Asian Thought). However, its mathematical resolution limit renders it blind to the nuanced schools and theoretical revolutions that drive philosophical debate.
- **Infomap is a cartographer of intellectual flow:** By tracking the diffusion of random walkers and compressing the information footprint of ideas, the Map Equation reconstructs the living contours of philosophy—isolating Continental Rationalism from British Empiricism, separating Socratic philosophy from Hellenistic practical wisdom, and highlighting the vital role of bridge thinkers.

In the history of ideas, dialogue and flow matter far more than static density. For understanding human thought across the centuries, **the Map Equation is the superior lens.**

---

*Written by Romeo Sofia – DTU Course 02805 Social Graphs & Interactions.*
