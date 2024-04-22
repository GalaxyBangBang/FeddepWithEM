import networkx as nx

G = nx.Graph()
e = (1, 2)
G.add_edge(*e)
e = (1, 3)
G.add_edge(*e)
print(G.number_of_nodes())