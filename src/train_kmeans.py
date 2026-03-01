import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import os

def run_kmeans_analysis(k_list=[100, 500, 1000]):
    # Load normalized patches
    data = torch.load('results/patches.pt').numpy().astype('float32')
    
    for k in k_list:
        print(f"--- Training K-Means with K={k} ---")
        # MiniBatchKMeans is much faster and stable on Mac
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2048, n_init=3, random_state=42)
        kmeans.fit(data)
        
        # Save centroids (interpretable strokes/dictionary)
        centroids = kmeans.cluster_centers_
        torch.save(torch.from_numpy(centroids), f'results/centroids_k{k}.pt')
        
        # Visualization: Create a 10x10 grid of the first 100 clusters
        plt.figure(figsize=(10, 10))
        for i in range(min(100, k)):
            plt.subplot(10, 10, i + 1)
            plt.imshow(centroids[i].reshape(5, 5), cmap='gray')
            plt.axis('off')
        plt.suptitle(f"Learned Visual Dictionary (K={k})")
        plt.savefig(f'results/dictionary_k{k}.png')
        plt.close()
        print(f"✅ Finished K={k}. Check results/dictionary_k{k}.png")

if __name__ == "__main__":
    if not os.path.exists('results'): os.makedirs('results')
    run_kmeans_analysis()