import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import os
import cv2
import maxflow

import visualize_data


def read_data(path: str) -> np.ndarray:
    """
    Function to read the image called 'nerves_part.tiff' from a given path

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
    

if __name__ == '__main__':
    data = read_data("data")

    # visualize_data.visualize(data[0], title="First slice of the image")
    # visualize_data.visualize_multiple(data[:10], size=2, title="First 10 slices")
    
    mu1, mu2 = visualize_data.visualize_histogram(data[0]) #prior knowledge
    
    print(mu1, mu2)
   