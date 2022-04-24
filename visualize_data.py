import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from typing import Tuple, Union


sns.set_theme(palette=sns.color_palette("husl"))


def visualize(img: np.ndarray, size: int = 8, title: str = "") -> None:
    """
    Visualize a single slice.

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


def visualize_multiple(
    img_arr: np.ndarray, size: int = 8, title: str = "", columns: int = 4
) -> None:
    """
    Display multiple slices, expanding to multiple rows if needed.

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

    fig, ax_arr = plt.subplots(
        num_rows, num_columns, figsize=(size * num_rows, size * num_columns)
    )

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


def visualize_histogram(
    data: np.ndarray,
    size: int = 8,
    title: str = "Slices $\mu$'s",
    bins: int = 100,
    color: Union[str, Tuple[int, int, int]] = "black",
    edge_color: Union[str, Tuple[int, int, int]] = "white",
    th: float = 0.5,
    save_path: Union[None, str] = None,
    display: bool = True
) -> Tuple[int, int]:
    """
    Visualize the histogram, with the input image along side
    a masked image using the threshold and get the average
    below and above the input threshold.

    Arguments
    ---------
    data: np.ndarray
        Slices to analyze
    size: int = 8
        Size of the whole figure
    title: str = "Slices $\mu$'s"
        Title of the figure
    bins: int = 100
        Number of bins of the histogram
    color: Union[str, Tuple[int, int, int]] = "black"
        Color of the histogram (follow plt colors)
    edge_color: Union[str, Tuple[int, int, int]] = "white"
        Color of the edges of the histogram (follow plt colors)
    th: float = 0.5
        Threshold to do the division
    save_path: Union[None, str] = None
        Path to save the image to. If it is None, it will not save it
    display: bool = True
        Show or not the plot

    Returns
    -------
    mu1: int
        Average below the threshold
    mu2: int
        Average above the threshold

    """

    data_th = np.zeros(np.shape(data))
    data_th[data > th] = 1

    mu1 = np.mean(data[data_th == 0])
    mu2 = np.mean(data[data_th == 1])

    edges = np.linspace(0, 1, bins)

    fig = plt.figure(figsize=(size, size))

    fig.suptitle(title)

    gs = GridSpec(nrows=2, ncols=2)

    # Original image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(data[0], cmap="gray")
    ax0.set_title("Original Image")

    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.grid(False)

    # Thresholded image
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(data_th[0], cmap="gray")
    ax1.set_title(f"Thresholded Image (threshold = {th})")

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.grid(False)

    # Histogram of the original image
    ax2 = fig.add_subplot(gs[1, :])
    ax2.hist(data.ravel(), edges, color=color, edgecolor=edge_color)
    ax2.set_xlabel("Pixel values")
    ax2.set_ylabel("Count")
    ax2.set_title("Intensity histogram")

    ax2.axvline(th, color=sns.color_palette()[0], label="Threshold")

    ax2.axvline(mu1, color=sns.color_palette()[1], label=f"$\mu_1$: {mu1:.3f}")
    ax2.axvline(mu2, color=sns.color_palette()[2], label=f"$\mu_2$: {mu2:.3f}")

    ax2.legend()

    if save_path != None:
        plt.savefig(save_path)

    if display:
        plt.show()
    else:
        plt.close()

    return mu1, mu2


def display_img_snake(
    img: np.ndarray, snake: np.ndarray, size: int = 8, title: str = ""
):
    """
    Displays both the image and a single snake.

    Arguments
    ---------
    img: np.ndarray
        Slice to display
    snake: np.ndarray
        Snake to plot
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
    snake = np.vstack([snake, snake[0]])
    plt.plot(snake[:, 1], snake[:, 0])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(title)
    plt.show()

def display_img_multiple_snake(
    img: np.ndarray, snakes: list, size: int = 8, title: str = ""
):
    """
    Displays both the image and a single snake.

    Arguments
    ---------
    img: np.ndarray
        Slice to display
    snakes: list
        List of snakes to plot
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
    for snake in snakes:
        snake = np.vstack([snake, snake[0]])
        plt.plot(snake[:, 1], snake[:, 0])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(title)
    plt.show()
