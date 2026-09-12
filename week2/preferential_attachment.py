import networkx as nx
import random
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_node(0)
G.add_node(1)
G.add_edge(0, 1)
a = [0,1];

for i in range(2,100):
    G.add_node(i)
    j = random.choice(a)    
    G.add_edge(i, j)
    a.append(i)
    a.append(j)

nx.draw(G, with_labels=True, node_color='lightblue')
plt.show()