# mnist-patch-dictionary-learning
Unsupervised dictionary learning and reconstruction analysis of MNIST handwritten digits using K-Means clustering on $5 \times 5$ image patches.
## 🚀 Project Overview
This project explores **Dictionary Learning** through unsupervised K-Means clustering on local image structures. By decomposing MNIST digits into $5 \times 5$ patches, we identify the fundamental "visual primitives" (edges, strokes, and curves) that constitute handwritten text.

## 🛠️ Key Features
* **Patch Extraction:** Efficient sliding window extraction generating over 20 million patches.
* **Preprocessing:** Background noise removal and **Unit Norm Normalization** for geometric consistency.
* **Scaling K:** Analysis of dictionary coherence as $K$ increases from 100 to 10,000+.
* **Image Reconstruction:** Rebuilding original digits using only the learned visual dictionary.

## 🔬 Mathematical Foundation
To focus on the structural patterns rather than pixel intensity, each patch vector $v$ is normalized to unit length. The similarity is then measured using Euclidean distance, which relates to cosine similarity as:
$$d(\mathbf{u}, \mathbf{v}) = \sqrt{2 - 2\langle\mathbf{u}, \mathbf{v}\rangle}$$

## 📂 Repository Structure
* `src/`: Core Python scripts for data processing and training.
* `notebooks/`: Visualization of clusters and reconstruction results.
* `results/`: Saved cluster centroids and performance plots.