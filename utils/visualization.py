import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_images(images):
    plt.figure(figsize=(16, 8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    plt.show()
