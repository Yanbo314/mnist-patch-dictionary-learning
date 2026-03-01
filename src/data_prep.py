import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import os

def extract_mnist_patches(num_images=20000, patch_size=5):
    """
    Extract and normalize 5x5 patches from MNIST dataset.
    """
    # Load MNIST training data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    patch_list = []
    print(f"Starting extraction from {num_images} images...")

    for i, (images, _) in enumerate(tqdm(loader)):
        if i * 128 >= num_images:
            break
        
        # Sliding window extraction using unfold
        # Shape: [Batch, 1, 28, 28] -> [Batch, 1, 24, 24, 5, 5]
        patches = images.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        
        # Reshape to [N, 25] (where 25 = 5 * 5)
        patches = patches.contiguous().view(-1, patch_size * patch_size)
        
        # Filter out near-blank patches (sum of pixels > 0.1)
        mask = patches.sum(dim=1) > 0.1
        active_patches = patches[mask]
        
        # Unit Norm Normalization: v = v / ||v||
        # Essential for measuring shape similarity regardless of intensity
        norm = torch.norm(active_patches, p=2, dim=1, keepdim=True)
        active_patches = active_patches / (norm + 1e-8)
        
        patch_list.append(active_patches)
            
    # Concatenate all batches into a single tensor
    final_data = torch.cat(patch_list, dim=0)
    print(f"Extraction finished. Total valid patches: {final_data.shape[0]}")
    return final_data

if __name__ == "__main__":
    # Run extraction and save tensor for training
    if not os.path.exists('results'):
        os.makedirs('results')
    
    data_tensor = extract_mnist_patches(num_images=10000)
    torch.save(data_tensor, 'results/patches.pt')
    print("Tensor saved to results/patches.pt")