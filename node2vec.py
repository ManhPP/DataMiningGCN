import stellargraph as sg
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

from stellargraph.data import BiasedRandomWalk # Second-order random walk
from stellargraph import StellarGraph
from stellargraph import datasets

# load dataset
dataset = datasets.Cora()
# display(HTML(dataset.description))

# G is the whole graph and info
# node_subjects is label of documents
G, node_subjects = dataset.load(largest_connected_component_only=True)
rw = BiasedRandomWalk(G)

walks = rw.run(
    nodes=list(G.nodes()),  # root nodes
    length=100,  # maximum length of a random walk
    n=10,  # number of random walks per root node
    p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
)
print("Number of random walks: {}".format(len(walks)))

from gensim.models import Word2Vec

# As gensim requires string input, we have to convert node index into string
str_walks = [[str(n) for n in walk] for walk in walks]
model = Word2Vec(str_walks, 
                 sg=0, # 0 : CBOW, 1: skip-gram 
                 size=128, window=5, min_count=0, sg=1, workers=2, iter=1)

# Retrieve node embeddings and corresponding subjects
node_ids = model.wv.index2word  # list of node IDs
node_embeddings = (
    model.wv.vectors
)  # numpy.ndarray of size number of nodes times embeddings dimensionality
node_targets = node_subjects[[int(node_id) for node_id in node_ids]]

# X will hold the 128-dimensional input features
X = node_embeddings
# y holds the corresponding target values
y = np.array(node_targets)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, test_size=None)

clf = LogisticRegressionCV(
    Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=300
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)