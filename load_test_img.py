from data_preprocessing import XrdDataset  
import matplotlib.pyplot as plt
import numpy as np
from plot import plot_reconstruction


EXPERIMENTS = {
    422: 'mfxl1025422',
    522: 'mfxl1027522'
}

def run():
    data_dir = 'data'
    data_id = EXPERIMENTS[422]
    dataset = XrdDataset(data_dir, data_id, rescale=True, apply_pooling=True)
    print(f"Number of images: {len(dataset)}")
    index = [np.random.randint(0, len(dataset)) for _ in range(5)]
    for j, i in enumerate(index):
        plot_reconstruction(dataset[i], dataset[i], idx=i)
        # img = dataset[i]
        # print(f"Image {i} shape: {img.shape}")
        # print(f"Image {i} mean: {img.mean()}")
        # print(f"Image {i} std: {img.std()}")
        # print(f"Image {i} min: {img.min()}")
        # print(f"Image {i} max: {img.max()}")
        # print(f"Image {i} dtype: {img.dtype}")
        # plt.imshow(img[0].cpu().numpy(), cmap='gray')  # Display image in grayscale
        # plt.savefig(f'img_{i}.png')  # Save the image

        plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    run()