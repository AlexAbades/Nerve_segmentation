import preprocess_data
import visualize_data
import MRF
import deformable_models


import numpy as np

if __name__ == "__main__":
    # Process data
    data = preprocess_data.read_data("data")[:20]

    # Visualize intensity histogram
    mu1, mu2 = visualize_data.visualize_histogram(
        data, save_path="outputs/plots/histogram", th=0.4, bins=70, display=False
    )

    # Binary segmentation with MRF
    data_result = MRF.graph_3d_segmetation(data, mu1, mu2, 0.001)
    # Visualize segmentation of first image
    # visualize_data.visualize(data_result[0])

    circles_center = [
        (150, 39),
        (132, 106),
        (117, 282),
        (216, 168),
        (258, 131)

    ]
    circles_r = [
        16,
        12,
        12,
        17,
        18
    ]

    # Volumetric segmentation of few nerves using Deformable models
    snakes = deformable_models.create_multiple_circles(circles_center, circles_r, 100)
    

    # visualize_data.display_img_multiple_snake(data_result[0], snakes)

    for i in range(1, 31):
        tau = 1
        for j in range(len(snakes)):
            snakes[j] = deformable_models.deformable(data_result[0], snakes[j], offside_out=5, tau=tau)


        if i % 10 == 0:
            tau /= 2
            visualize_data.display_img_multiple_snake(data_result[0], snakes)

    # print(deformable_models.get_snake_normals(snake))


