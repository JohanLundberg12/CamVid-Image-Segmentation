"""
For a given model, we compute and concatenate all its outputs on the validation set examples
as one long vector summmarizing where it stands in "function space". We get one such vector
for each epoch. This allows us to plot many learning trajectories, one for each initialization seed,
with or without pre-training. Using a dimensionality reduction algorithm (T-SNE) we can then visualize
these vectors in 2d. 
See figure 15.1 in [Goodfellow] or "Why Does Unsupervised Pre-training Help Deep Learning?" section 6.3. (3 pages)
"""
import numpy.typing as npt

import sys
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_model_trajectories(m1_preds:npt.NDArray, m2_preds:npt.NDArray):
    tsne = TSNE(n_components=2)
    cmap = ['b', 'g', 'r', 'c', 'm', 'y', 'b', 'w']

    for epoch, (preds1, preds2) in enumerate(zip(m1_preds, m2_preds)):
        m1_epoch_trajectory = tsne.fit_transform(preds1)
        m2_epoch_trajectory = tsne.fit_transform(preds2)

        plt.scatter(m1_epoch_trajectory[:, 0], m1_epoch_trajectory[:, 1], c=cmap[epoch], marker="^")
        plt.scatter(m2_epoch_trajectory[:, 0], m2_epoch_trajectory[:, 1], c=cmap[epoch], marker="x")

    plt.show()



if __name__ == "__main__":
    pass
    


    
