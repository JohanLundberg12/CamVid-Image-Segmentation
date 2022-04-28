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

def visualize_model_trajectories(m1_preds:np.ndarray, m2_preds:np.ndarray):
    tsne = TSNE(n_components=2, init="random", random_state=42)
    m1_trajectory = list()
    m2_trajectory = list()

    for epoch, (preds1, preds2) in enumerate(zip(m1_preds, m2_preds)):
        m1_epoch_trajectory = tsne.fit_transform(preds1)
        m2_epoch_trajectory = tsne.fit_transform(preds2)

        m1_trajectory.append(m1_epoch_trajectory)
        m2_trajectory.append(m2_epoch_trajectory)

    colors = ['red', 'blue']

    for i, (epoch_trajectory_1, epoch_trajectory_2) in enumerate(zip(m1_trajectory, m2_trajectory)):
        plt.scatter(epoch_trajectory_1[:, 0], epoch_trajectory_1[:, 1], c=colors[i], marker="o")
        plt.scatter(epoch_trajectory_2[:, 0], epoch_trajectory_2[:, 1], c=colors[i], marker="^")

    plt.savefig("test.png")



if __name__ == "__main__":
    file1, file2 = sys.argv[1], sys.argv[2]
    preds1, preds2 = json.loads(file1), json.loads(file2)
    


    
