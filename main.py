from turtle import color
import preprocess_data
import visualize_data
import MRF
import deformable_models
import analyze_data
from PIL import Image
import seaborn as sns
from tqdm import tqdm
import tiffile


import os
import cv2


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
        data, th=0.36, bins=50, display=True
    )
    

    # Save images
    # for i, img in enumerate(tqdm(data)):  
    #     visualize_data.visualize(img, save_path=f"outputs/original/sample_{i}", display=False)  

    data_shape = data.shape

    # Binary segmentation with MRF
    data_result = np.empty((0, data_shape[1], data_shape[2]))
    for i in range(0, len(data), 100):
        data_result = np.append(data_result, MRF.graph_3d_segmetation(data[i:i+100], mu1, mu2, 0.001), axis=0)
    
    # Data results dimensions 
    # 1024, 350 , 350 -> Range(0,1) 
    
    data_result_save = data_result*255
    data_result_save = data_result_save.astype(np.uint8)


    print(data_result_save.shape) 
    
    #tiffile.imwrite('segmentation_MRF0.tiff', data_result_save)



    slice0=data_result[0]
    print('Myelin density')
    print((1-(np.sum(slice0)/(350*350)))*100)
    #Myelin density from slice 0: 39.06285714285714
    
    

    # Save data result 
    # for i, img in enumerate(tqdm(data_result)):  
    #     visualize_data.visualize(img, save_path=f"outputs/segmentation/sample_{i}", display=False)  

    circles_center = [(150, 39), (117, 282), (216, 168), (258, 131), (209, 214), (281, 230), (47, 70)]
    circles_r = [16, 12, 17, 18, 10, 10, 8]
    max_distances = [10, 15, 10, 10, 10, 10, 10]

    colors = sns.color_palette("bright", len(circles_center))

    bigger_circles_r = [x + d for x, d in zip(circles_r, max_distances)]


    # Volumetric segmentation of few nerves using Deformable models
    initial_snakes = deformable_models.create_multiple_circles(circles_center, circles_r, 100)

    bigger_initial_snakes = deformable_models.create_multiple_circles(circles_center, bigger_circles_r, 100)

    all_snakes, all_snakes_big = deformable_models.extrapolate_volum(data_result[:2], initial_snakes, bigger_initial_snakes, max_distance=max_distances)
    
    print(all_snakes.shape)
    print(all_snakes.shape)

    #Save volum:
    nerve_segmentation=visualize_data.save_volume(all_snakes,all_snakes_big,data[:2])

    # all_snakes, all_snakes_big = deformable_models.extrapolate_volum(data_result, initial_snakes, bigger_initial_snakes, max_distance=max_distances)

    #Extract Radius and Areas:
    mean_radi_axon,mean_area_axon,mean_radi_nerve,mean_area_nerve,mean_radi_myelin_thickness,mean_area_myelin_thickness=analyze_data.compute_average_in_out_radius_area(all_snakes, all_snakes_big)
    print('Mean radi axon:')
    print(mean_radi_axon)
    print(np.mean(mean_radi_axon))
    print('Mean area axon')
    print(mean_area_axon)
    print(np.mean(mean_area_axon))
    print('Mean radi nerve')
    print(mean_radi_nerve)
    print(np.mean(mean_radi_nerve))
    print('Mean area nerve')
    print(mean_area_nerve)
    print(np.mean(mean_area_nerve))
    print('Mean radi myelin thickness')
    print(mean_radi_myelin_thickness)
    print(np.mean(mean_radi_myelin_thickness))
    print('Mean area myelin thickness')
    print(mean_area_myelin_thickness)
    print(np.mean(mean_area_myelin_thickness))


    visualize_data.save_volume(all_snakes, data)


    # visualize_data.save_volume(all_snakes, data)
    #for i in tqdm(range(len(data_result))):
        #snakes = np.array(list(all_snakes[i]) + list(all_snakes_big[i]))
        #visualize_data.display_img_multiple_snake(data[i], snakes, colors=colors, display=False, save_path=f"outputs/nerves/sample_{i}")

    

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

    """
    img_path = "data/sample_100.png"
    data = np.array(cv2.imreadmulti(img_path,  flags=cv2.IMREAD_GRAYSCALE)[1])
    data = data.astype(float) / 255

    mu1, mu2 = visualize_data.visualize_histogram(
        data, th=0.45, bins=50, display=False
    )

    data_result = MRF.graph_3d_segmetation(data, mu1, mu2, 0.01)

    snake_in = deformable_models.create_multiple_circles([(50, 50)], [18], 100)
    snake_out = deformable_models.create_multiple_circles([(50, 50)], [45], 100)

    snake_in, snake_out = deformable_models.extrapolate_volum(data_result, snake_in, snake_out, [20], iter_first=50)

    visualize_data.plot_inside_ouside_masks(data[0], snake_in[0, 0], snake_out[0, 0])

    # print(snake_in)

    # visualize_data.display_img_snake(data_result[0], snake_in[0, 0])
    # visualize_data.display_img_snake(data_result[0], snake_out[0, 0])
    """