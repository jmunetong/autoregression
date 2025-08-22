import os 

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_preprocessing import XrdDataset
from argparse import ArgumentParser
from plot import plot_single_image, transform_to_image


DATA_PATH = "data"

EXPERIMENTS = {
    422: 'mfxl1025422',
    522: 'mfxl1027522'
}

SAVE_DIR = 'test_files/sample_images'

def main(args):
        # Dataset and Dataloader
    dataset = XrdDataset(data_dir=args.data_path,apply_pooling=args.avg_pooling, data_id=EXPERIMENTS[args.data_id], top_k=args.topk)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    batch  = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    for idx, image in enumerate(batch):
        image = image.permute(1, 2, 0)  # Remove batch dimension if present
        image = transform_to_image(image)
        plot_single_image(image, idx, SAVE_DIR)
    print(f"Images saved to {SAVE_DIR}")

    
      


if __name__ == "__main__":
    parser = ArgumentParser(description="Image Analysis Script")
       # Data parameters
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the data directory")
    parser.add_argument("--data_id", type=int, default=522, choices=[422, 522], help="Experiment number")
    parser.add_argument("--avg_pooling", action='store_true', help="Apply average pooling to the images")
    parser.add_argument("--topk", type=float, default=1.0, help="Top k percent of images to use for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch siz")

    args = parser.parse_args()
    main(args)



