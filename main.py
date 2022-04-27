from turtle import color
import preprocess_data
import visualize_data
import MRF
import deformable_models
from PIL import Image



import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Process data
    """ 
    data_batch1 = preprocess_data.read_data("data")[:100]
    data_batch2 = preprocess_data.read_data("data")[100:200]
    data_batch3 = preprocess_data.read_data("data")[200:300]
    data_batch4 = preprocess_data.read_data("data")[300:400]

    visualize_data.visualize(data_batch1[0], save_path="outputs/original/sample_0", display=False)
    visualize_data.visualize(data_batch2[0], save_path="outputs/original/sample_100", display=False)
    visualize_data.visualize(data_batch3[0], save_path="outputs/original/sample_200", display=False)
    visualize_data.visualize(data_batch4[0], save_path="outputs/original/sample_300", display=False)

    mu1, mu2 = visualize_data.visualize_histogram(
        data_batch1, save_path="outputs/plots/histogram", th=0.36, bins=50, display=False
    )

    data_result = MRF.graph_3d_segmetation(data_batch1, mu1, mu2, 0.001)
    visualize_data.visualize(data_result[0], save_path="outputs/segmentation/sample_0", display=False)

    data_result = MRF.graph_3d_segmetation(data_batch2, mu1, mu2, 0.001)
    visualize_data.visualize(data_result[0], save_path="outputs/segmentation/sample_100", display=False)

    data_result = MRF.graph_3d_segmetation(data_batch3, mu1, mu2, 0.001)
    visualize_data.visualize(data_result[0], save_path="outputs/segmentation/sample_200", display=False)

    data_result = MRF.graph_3d_segmetation(data_batch4, mu1, mu2, 0.001)
    visualize_data.visualize(data_result[0], save_path="outputs/segmentation/sample_400", display=False)
    """

    
    data = preprocess_data.read_data("data")[:10]

    # Visualize intensity histogram
    mu1, mu2 = visualize_data.visualize_histogram(
        data, save_path="outputs/plots/histogram", th=0.36, bins=50, display=False
    )

    # Binary segmentation with MRF
    data_result = MRF.graph_3d_segmetation(data, mu1, mu2, 0.001)
    # Visualize segmentation of first image
    # visualize_data.visualize(data_result[0])

    circles_center = [(150, 39), (132, 106), (117, 282), (216, 168), (258, 131)]
    circles_r = [16, 12, 12, 17, 18]

    # Volumetric segmentation of few nerves using Deformable models
    initial_snakes = deformable_models.create_multiple_circles(circles_center, circles_r, 100)


    all_snakes = deformable_models.extrapolate_volum(data_result, initial_snakes)

    print(all_snakes.shape)

    visualize_data.save_volume(all_snakes, data)


    # data_post = np.array(data_post)
    # vol_data = Image.fromarray(data_post)
    # vol_data.save('test_volume.tif')

    # for i, snakes in enumerate(all_snakes):
    #     visualize_data.display_img_multiple_snake(data[i], snakes)


    # # visualize_data.display_img_multiple_snake(data_result[0], snakes)

    # for i in range(1, 31):
    #     tau = 1
    #     for j in range(len(snakes)):
    #         snakes[j] = deformable_models.deformable(data_result[0], initial_snakes[j], offside_out=5, tau=tau)

    #     if i % 10 == 0:
    #         tau /= 2
    #         visualize_data.display_img_multiple_snake(data[0], snakes)

    # print(deformable_models.get_snake_normals(snake))
