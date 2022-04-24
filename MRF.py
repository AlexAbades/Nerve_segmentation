import numpy as np
import maxflow
from visualize_data import visualize

import visualize_data


def likelihood(D, S, *mus):
    """
    Using Markov Random fields we calclate the likelihood as one clique
    potentials.
    We calculate the likelihood as the squared difference from each pixel
    to the mean. That means, if we have a initial configuration where class 1
    has a value of 170, and the pixel we are analysing belongs to that class,
    the one clique potential for that pixel will be 170 - value of that pixel.
    Value of a pixel it's its intensity.
    V_1(f_i) = (mu(f_i)-d_i)^2
    U(d|f) = SUM(V_1(f_i))

    Parameters
    ----------
    D : ndarray
        Image we want to threshold
    S : ndarray
        Site image, labeled image depending on threshold.
    *mus : Tuple
        Mean values for the labels of the site Image. Have to be in labeling
        order. If Site image has 3 differet labels, we need 3 different mus.
        where mu_0 is queal to the value of the label 0 on the site image

    Returns
    -------
    V1 : Scalar
        Likelihood Energy

    """
    # check that we have euqal mus as labels
    if len(np.unique(S)) != len(mus):
        print("Same number of mus must be given as number of labels in Site image")
        print("Labels of site Image: ", np.unique(S))
        print("Mus provided: ", mus)
        return

    # To avoid coressed reference:
    I = np.zeros(S.shape)

    for mu, i in zip(mus, range(len(mus))):
        I[S == i] = mu

    Dif = (D - I) ** 2
    V1 = sum(sum(Dif))

    return V1, Dif


def priorP(S: np.array, beta: int):
    """
    We define the 2 clique potentials for discrete labels which penalizes
    neighbouring labels being different. With a 4 Neighbour configuration, (+)

    V_2(f_i, f_i') = 0 if (f_i = f_i'); beta otherwise
    U(f) = SUM(V2(f_i,f_i')) for i~i'

    Parameters
    ----------
    S : np.array
        Site image, labeled image depending on threshold.
    beta : int
        Smothness weight .

    Returns
    -------
    V2: int
        Prior Potential.

    """

    # Check if the pixel it's the same as the neighbour to the right and left
    S_col = S[1:, :] != S[:-1, :]
    # Check if the pixel it's the same as the upper and lower neighbour
    S_row = S[:, 1:] != S[:, :-1]

    # Total amount of different pixels
    total_different = S_col.sum() + S_row.sum()
    V2 = beta * total_different

    return V2


def graph_creation(img: np.ndarray, mu1: float, mu2: float, beta: int = 0.01):
    # Create the graph.
    g = maxflow.Graph[float]()

    # Add the nodes. nodeids has the identifiers of the nodes in the grid. Note that nodeids.shape == img.shape
    nodeids = g.add_grid_nodes(img.shape)
    g.add_grid_edges(nodeids, beta)

    # Add the terminal edges. The image pixels are the capacities of the edges from the source node. The inverted image pixels
    # are the capacities of the edges to the sink node.
    # Check if we have to give the negative of the image ~I
    w_t = (img - mu1) ** 2  # node-t weight
    w_s = (img - mu2) ** 2  # node-s weight
    g.add_grid_tedges(nodeids, w_t, w_s)

    # Find the maximum flow.
    g.maxflow()

    # Get the segments of the nodes in the grid.
    # sgm.shape == nodeids.shape
    sgm = g.get_grid_segments(nodeids)

    # The labels should be 1 where sgm is False and 0 otherwise.
    img_result = np.logical_not(sgm).astype(int)

    # visualize_data.visualize(img_result)
    return img_result


def graph_allvolum(data, mu1, mu2):
    all_volume = []
    for img in data:
        img_g = graph_creation(img, mu1, mu2)
        all_volume.append(img_g)

    # visualize_data.visualize(all_volume[0])
    return all_volume


def graph_3d_segmetation(
    data: np.ndarray, mu1: float, mu2: float, beta: float, z_increment: float = 1
) -> np.ndarray:
    """
    Creates and solves a 3D maxflow Graph and generates the links with the pixel to the
    up, down left and right in each image, and the connections with the pixel from
    the previous and following image. It returns the data segmented in two classes.

    Arguments
    ---------
    data: np.ndarray
        Group of images
    mu1: float
        Mu belonging to the first class
    mu2: float
        Mu belonging to the second class
    beta: float
        Beta to use in the graph
    z_increment: float = 1
        Value to increment the connection along the z axis

    Returns
    -------
    data_result: np.ndarray
        Segmented data using the maxflow graph
    """

    # Create the graph
    g = maxflow.Graph[float]()

    # We need to add a structure that indicates how the nodes of the graph must be connected.
    # As the data is in 3D, we have to make edges for the third dimension too.
    # We may be interested in changing the weight of the edges from the z direction.
    # 3D structure to create the links
    structure = maxflow.vonNeumann_structure(ndim=3, directed=True)
    # Change weight of the connections to the 'following image' by a factor.
    # We only change the connection with the following, there is no need for doing it for the previous one.
    structure[2, 1, 1] = z_increment

    # Add the nodes. nodeids has the identifiers of the nodes in the grid. Note that nodeids.shape == img.shape
    nodeids = g.add_grid_nodes(data.shape)
    g.add_grid_edges(nodeids, beta)

    # Add the terminal edges. The image pixels are the capacities of the edges from the source node. The inverted image pixels
    # are the capacities of the edges to the sink node.
    # Check if we have to give the negative of the image ~I
    w_t = (data - mu1) ** 2  # node-t weight
    w_s = (data - mu2) ** 2  # node-s weight
    g.add_grid_tedges(nodeids, w_t, w_s)

    # Find the maximum flow.
    g.maxflow()

    # Get the segments of the nodes in the grid.
    # sgm.shape == nodeids.shape
    sgm = g.get_grid_segments(nodeids)

    # The labels should be 1 where sgm is False and 0 otherwise.
    data_result = np.logical_not(sgm).astype(int)

    # visualize_data.visualize(img_result)
    return data_result
