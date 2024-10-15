import matplotlib.pyplot as plt

def show_image(image_tensor, title=None):
    """Helper function to visualize a single image."""
    image = image_tensor.cpu().detach().numpy()
    image = image.transpose(1, 2, 0)  # Convert to HWC format for display
    plt.imshow((image * 0.5 + 0.5))  # Normalize to [0, 1] for display
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
