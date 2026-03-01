import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

def test_reconstruction(k=1000):
    # 1. Load the dictionary we just trained
    dict_path = f'results/centroids_k{k}.pt'
    if not os.path.exists(dict_path):
        print(f"Error: Dictionary {dict_path} not found. Run train_kmeans.py first.")
        return
    
    dictionary = torch.load(dict_path).float()
    
    # 2. Get a fresh image from MNIST test set
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    image, _ = test_set[0] # Take the first image (a '7')
    img_data = image.squeeze() # 28x28
    
    # 3. Reconstruct using 5x5 non-overlapping blocks
    reconstructed = torch.zeros_like(img_data)
    
    for i in range(0, 25, 5): # Simple grid-based reconstruction
        for j in range(0, 25, 5):
            patch = img_data[i:i+5, j:j+5].flatten()
            
            if patch.sum() > 0.1: # Skip blank areas
                # Normalize patch to match dictionary style
                norm = torch.norm(patch, p=2)
                normalized_patch = patch / (norm + 1e-8)
                
                # Find the nearest neighbor word in our dictionary
                distances = torch.norm(dictionary - normalized_patch, dim=1)
                best_match_idx = torch.argmin(distances)
                
                # Replace with the dictionary word, scaled back to original intensity
                word = dictionary[best_match_idx].reshape(5, 5)
                reconstructed[i:i+5, j:j+5] = word * norm
                
    # 4. Save the comparison result
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original MNIST Digit")
    plt.imshow(img_data, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Reconstructed (K={k})")
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')
    
    plt.savefig(f'results/comparison_k{k}.png')
    print(f"🎉 Comparison saved! Open results/comparison_k{k}.png to see the magic.")

if __name__ == "__main__":
    test_reconstruction(k=1000)