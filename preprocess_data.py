import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import os
import cv2
import maxflow

import visualize_data
import MRF


def read_data(path: str) -> np.ndarray:
    """
    Function to read the image called 'nerves_part.tiff' from a given path.

    Arguments
    ---------
    path: str
        Path where the image is located

    Returns
    -------
    data: np.ndarray
        Volumetric image

    """

    img_path = os.path.join(path, "nerves_part.tiff")
    data = np.array(cv2.imreadmulti(img_path)[1])
    data = data.astype(float) / 255

    return data
