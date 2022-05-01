import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from skimage.draw import polygon2mask
import tiffile

from typing import Tuple, Union

# from voxelfuse.voxel_model import VoxelModel
# from voxelfuse.mesh import Mesh
# from voxelfuse.primitives import generateMaterials


sns.set_theme(palette=sns.color_palette("husl"))


def visualize(img: np.ndarray, size: int = 8, title: str = "", save_path: Union[None, str] = None, display: bool = True) -> None:
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
    save_path: Union[None, str] = None
        Path to save the image to. If it is None, it will not save it
    display: bool = True
        Show or not the plot

    Returns
    -------
    None

    """
    plt.figure(figsize=(size, size), frameon=False)
    plt.imshow(img, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(title)
    
    if save_path != None:
        plt.savefig(save_path)

    if display:
        plt.show()
    else:
        plt.close()


def visualize_multiple(
    img_arr: np.ndarray, size: int = 8, title: str = "", columns: int = 4, save_path: Union[None, str] = None, display: bool = True
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
    save_path: Union[None, str] = None
        Path to save the image to. If it is None, it will not save it
    display: bool = True
        Show or not the plot

    columns: int = 4
        Number of columns for the data
    """

    num_columns = min(columns, len(img_arr))

    num_rows = int(np.ceil(len(img_arr) / num_columns))

    fig, ax_arr = plt.subplots(
        num_rows, num_columns, figsize=(size * num_rows, size * num_columns), frameon=False
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

    if save_path != None:
        plt.savefig(save_path)

    if display:
        plt.show()
    else:
        plt.close()


def visualize_histogram(
    data: np.ndarray,
    size: int = 8,
    title: str = "Slices $\mu$'s",
    bins: int = 100,
    color: Union[str, Tuple[int, int, int]] = "black",
    edge_color: Union[str, Tuple[int, int, int]] = "white",
    th: float = 0.5,
    save_path: Union[None, str] = None,
    display: bool = True,
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

    bin_space = np.arange(0, 1, 1 / bins)
    hist, bin_edges = np.histogram(data, bins=bin_space)
    center_bins = np.array(
        [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    )

    mu1 = center_bins[hist[center_bins <= th].argmax()]
    mu2 = center_bins[hist[center_bins > th].argmax() + len(hist[center_bins <= th])]

    fig = plt.figure(figsize=(size, size), frameon=False)

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
    ax2.bar(center_bins, hist, width=bin_space[1], color=color, edgecolor=edge_color)
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

    plt.figure(figsize=(size, size), frameon=False)
    plt.imshow(img, cmap="gray")
    snake = np.vstack([snake, snake[0]])
    plt.plot(snake[:, 1], snake[:, 0])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(title)
    plt.show()


def display_img_multiple_snake(
    img: np.ndarray, snakes: list, size: int = 8, title: str = "", save_path: Union[None, str] = None, display: bool = True, colors: Union[None, list] = None
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
    save_path: Union[None, str] = None
        Where to save the image
    display: bool = True
        Show or not the plot

    Returns
    -------
    None
    """

    plt.figure(figsize=(size, size), frameon=False)
    plt.imshow(img, cmap="gray")
    for i, snake in enumerate(snakes):
        snake = np.vstack([snake, snake[0]])

        if colors != None:
            color = colors[i % len(colors)]
            plt.plot(snake[:, 1], snake[:, 0], color=color)
        else:
            plt.plot(snake[:, 1], snake[:, 0])


    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(title)

    if save_path != None:
        plt.savefig(save_path)

    if display:
        plt.show()
    else:
        plt.close()
    

    
def save_volume(snakes_in, snakes_out, data)->None:
    """
    Given the interior and the exterior snakes, it creates a mask of Trues in between the snakes and it asigns a color to each snake. 

    ARGUMENTS
    ---------
    snakes_in:
    snakes_out: 
    data:

    RETURNS
    ---------
    None 
    """
    # Numpy vector to store the circles
    white_image = np.zeros(data.shape)

    # Color map to have different nerves in different colors 
    colors = np.round(np.linspace(150,255,7)).astype(np.uint8)
    

    # Get dimensions of the image 
    d, row, col = data.shape

    # snake(10,5,99,2)-> (slices, circles, points, dimensions_points)
    for s in range(d): # Iterate through each slide 
        for c in range(snakes_in.shape[1]): # Iterate through each circle
            mask_in = polygon2mask((row,col), snakes_in[s,c,:,:])
            mask_out = polygon2mask((row,col), snakes_out[s,c,:,:])
            mask = mask_out & ~mask_in
            idx = np.where(mask)
            white_image[s, idx[0], idx[1]] = colors[c]

    # Transform to uint 8
    white_image = white_image.astype(np.uint8)
    # Save
    tiffile.imwrite('snakes_segmentation.tiff', white_image)
    


    

    #model = VoxelModel(white_image, generateMaterials(4))  #4 is aluminium.
    #mesh = Mesh.fromVoxelModel(model)
    #mesh.export('mesh.stl')