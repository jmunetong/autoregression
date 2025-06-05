import torch
from diffusers import AutoencoderKL
import gc

def test_autoencoder_compatibility():
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running tests on {device}")

    # Test configurations: (in_channels, height, width)
    configurations = [
        # (1, 224, 224),
        # (3, 224, 224),
        # (1, 512, 512),
        # (3, 512, 512),
        # (4, 512, 512),
        # (3, 768, 512),
        # (5, 640, 480),
        # (3, 1024, 1024),
        (1, 834, 834),
        (1, 1667, 1665),
        (1, 1920, 1920),
        (1, 1664, 1664),
    ]

    print("\n{:<20} {:<10} {:<20} {:<20} {:<20}".format(
        "Configuration", "Status", "Input Shape", "Latent Shape", "Output Shape"))
    print("-" * 90)

    for in_channels, height, width in configurations:
        # Create the configuration string for display
        config_str = f"({in_channels}, {height}, {width})"

        try:
            # Create a fresh model with the correct in_channels
            vae = AutoencoderKL(
                in_channels=in_channels,        # Set input channels
                out_channels=in_channels,       # Match output channels with input
                down_block_types=("DownEncoderBlock2D",) * 4,
                up_block_types=("UpDecoderBlock2D",) * 4,
                block_out_channels=(64, 128, 256, 512),
                latent_channels=4,
                sample_size=64  # This doesn't limit input size
            )

            
            # Move model to device
            vae = vae.to(device)

            # Create random input with batch size 1
            input_tensor = torch.randn(1, in_channels, height, width).to(device)

            # Normalize to [-1, 1] as expected by VAE
            input_tensor = 2 * ((input_tensor - input_tensor.min()) /
                             (input_tensor.max() - input_tensor.min())) - 1

            # Process through VAE
            with torch.no_grad():
                # Encode
                latent = vae.encode(input_tensor).latent_dist.sample()

                # Decode
                output = vae.decode(latent).sample

            # Print results
            print("{:<20} {:<10} {:<20} {:<20} {:<20}".format(
                config_str,
                "✓ Success",
                str(tuple(input_tensor.shape)),
                str(tuple(latent.shape)),
                str(tuple(output.shape))
            ))

            # Check if output shape matches input shape
            if input_tensor.shape != output.shape:
                print(f"  Note: Input and output shapes don't match for {config_str}")

            # Calculate downsampling factor
            h_factor = input_tensor.shape[2] / latent.shape[2]
            w_factor = input_tensor.shape[3] / latent.shape[3]
            print(f"  Downsampling factor: {h_factor}x{w_factor}")

            # Free memory
            del vae, input_tensor, latent, output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        except Exception as e:
            # Print failure information
            print("{:<20} {:<10} {:<60}".format(
                config_str,
                "✗ Failed",
                str(e)[:60] + "..." if len(str(e)) > 60 else str(e)
            ))

            # Ensure memory is freed even on failure
            if 'vae' in locals():
                del vae
            if 'input_tensor' in locals():
                del input_tensor
            if 'latent' in locals():
                del latent
            if 'output' in locals():
                del output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

if __name__ == "__main__":
    test_autoencoder_compatibility()