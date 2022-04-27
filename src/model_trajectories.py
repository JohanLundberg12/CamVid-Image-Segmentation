"""
For a given model, we compute and concatenate all its outputs on the validation set examples
as one long vector summmarizing where it stands in "function space". We get one such vector
for each epoch. This allows us to plot many learning trajectories, one for each initialization seed,
with or without pre-training. Using a dimensionality reduction algorithm (T-SNE) we can then visualize
these vectors in 2d. 
See figure 15.1 in [Goodfellow] or "Why Does Unsupervised Pre-training Help Deep Learning?" section 6.3. (3 pages)
"""

import sys
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_model_trajectories(model_trajectories1, model_trajectories2):
    tsne = TSNE(n_components=2, init="random", random_state=42)

    model_trajectories1 = tsne.fit_transform(model_trajectories1)
    model_trajectories2 = tsne.fit_transform(model_trajectories2)

    plt.scatter(model_trajectories1[:,0], model_trajectories1[:,1], c="b")
    plt.scatter(model2[:,0], model1[:,1], c="r", label="")

    


if __name__ == "__main__":
    file1, file2 = sys.argv[1], sys.argv[2]

    preds1, preds2 = json.loads(file1), json.loads(file2)
    
