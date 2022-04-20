import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from typing import Tuple, Union


sns.set_theme(palette=sns.color_palette("husl"))

def visualize(img: np.ndarray, size: int = 8, title: str = "") -> None:
    """
    Visualize a single slice

    Arguments
    ---------
    img: np.ndarray
        Slice to display
    
    size: int = 8
        Size of the slice
    
    title: str = ""
        Optional title for the slice
    
    Returns
    -------
    None

    """
    plt.figure(figsize=(size, size))
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(title)
    plt.show()

def visualize_multiple(img_arr: np.ndarray, size: int = 8, title: str = "", columns: int = 4) -> None:
    """
    Display multiple slices, expanding to multiple rows if needed

    Arguments
    ---------
    img_arr: np.ndarray
        Multiple slices to display
    
    size: int = 8
        Size of the slice
    
    title: str = ""
        Optional title for the collection of slices
    
    columns: int = 4
        Number of columns for the data
    """

    num_columns = min(columns, len(img_arr))

    num_rows = int(np.ceil(len(img_arr) / num_columns))

    fig, ax_arr = plt.subplots(num_rows, num_columns, figsize = (size * num_rows, size * num_columns))

    fig.suptitle(title)

    # Will not use zip(img_arr, ax_arr) beacuse may be that some axes are not used, and therefore this way 
    # they will look bad, so first setup, then zip

    # Setup all the axes
    for ax in ax_arr.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    # Display the images
    for ax, img in zip(ax_arr.flatten(), img_arr):
        ax.imshow(img, cmap="gray")

    plt.show()

def visualize_histogram(img: np.ndarray, size: int = 8, title: str = "Slice Threshold",
                        bins: int = 100, color: Union[str, Tuple[int, int, int]] = "black",
                        edge_color: Union[str, Tuple[int, int, int]] = "white",
                        th: float = 0.36) -> Tuple[int, int]:
    """
    Visualize the histogram, with the input image along side
    a masked image using the threshold and get the average
    below and above the input threshold

    Arguments
    ---------
    img: np.ndarray
        Slice to analyze
    size: int = 8
        Size of the whole figure
    title: str = "Slice Threshold"
        Title of the figure
    bins: int = 100
        Number of bins of the histogram
    color: Union[str, Tuple[int, int, int]] = "black"
        Color of the histogram (follow plt colors)
    edge_color: Union[str, Tuple[int, int, int]] = "white"
        Color of the edges of the histogram (follow plt colors)
    th: float = 0.36
        Threshold to do the division

    Returns
    -------
    mu1: int 
        Average below the threshold
    mu2: int 
        Average above the threshold

    """


    img_th = np.zeros(np.shape(img))
    img_th[img > th] = 1

    mu1 = np.mean(img[img_th == 0])
    mu2 = np.mean(img[img_th == 1])
    
    edges = np.linspace(0, 1, bins)

    fig = plt.figure(figsize=(size, size))

    fig.suptitle(title)

    gs = GridSpec(nrows=2, ncols=2)
    
    # Original image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img, cmap="gray")
    ax0.set_title('Original Image')

    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.grid(False)

    # Thresholded image
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(img_th,cmap="gray")
    ax1.set_title(f'Thresholded Image (threshold = {th})')

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.grid(False)
    
    #Histogram of the original image
    ax2 = fig.add_subplot(gs[1, :])
    ax2.hist(img.ravel(), edges, color=color, edgecolor=edge_color)
    ax2.set_xlabel('Pixel values')
    ax2.set_ylabel('Count')
    ax2.set_title('Intensity histogram')

    ax2.axvline(th, color=sns.color_palette()[0], label="Threshold")

    ax2.axvline(mu1, color=sns.color_palette()[1], label="Below threshold average")
    ax2.axvline(mu2, color=sns.color_palette()[2], label="Above threshold average")
    
    ax2.legend()
    
    plt.show()

    return mu1, mu2