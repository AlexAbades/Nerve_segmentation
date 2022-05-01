
import numpy as np
import matplotlib.pyplot as plt


def compute_average_radius_area (snakes):
    """
    Function to compute the average radius and area of each circle

    Arguments
    ---------
    snakes np.ndarray

    Returns
    -------
    mean_radius: np.array
        mean radius for each circle
    mean_area: np.array
        mean area for each circle
    """
    radi_all_slices= np.zeros((len(snakes),snakes.shape[1]))
    for s in range(len(snakes)): 
        radi_circles=[]
        for c in range(snakes.shape[1]):
            center_x=np.mean(snakes[s,c,:,0])
            center_y=np.mean(snakes[s,c,:,1])
            
            diff_x=[]
            diff_y=[]

            for k,kk in enumerate(snakes[s,c,:,0]):
                diff_x.append(np.abs(center_x-snakes[s,c,k,0]))
                diff_y.append(np.abs(center_y-snakes[s,c,k,1]))

            radi_x= np.mean(diff_x)
            radi_y= np.mean(diff_y)
            radi= np.sqrt(radi_x**2+radi_y**2)
            radi_circles.append(radi)

        radi_all_slices[s,:]=np.array(radi_circles)

    mean_radius=  np.mean(radi_all_slices,axis=0) 
    mean_area= np.pi*(mean_radius**2)

    return mean_radius,mean_area


def compute_average_in_out_radius_area (snakes_in,snakes_out):
    """
    Function to compute the average radi and area of the axon, the nerve and the myelin_thickness.

    Arguments
    ---------
    snakes_in: np.ndarray
    snakes_out: np.ndarray  

    Returns
    -------
    radi:
    mean_radi_axon=mean_radius_in: np.list
    mean_radi_nerve=mean_radius_out: np.list
    mean_radi_myelin_thickness=difference of mean_radius_out-mean_radius_in: np.list

    area:
    mean_area_axon: np.list
    mean_area_nerve=np.list
    mean_area_myelin_thickness=difference of mean_area_nerve-mean_area_axon: np.list

    """
    mean_radi_axon,mean_area_axon=compute_average_radius_area(snakes_in)
    mean_radi_nerve,mean_area_nerve=compute_average_radius_area(snakes_out)
    mean_radi_myelin_thickness=[]
    mean_area_myelin_thickness=[]

    for i in range(len(mean_radi_nerve)):
        mean_radi_myelin_thickness.append(mean_radi_nerve[i]-mean_radi_axon[i])
        mean_area_myelin_thickness.append(mean_area_nerve[i]-mean_area_axon[i])

    return mean_radi_axon,mean_area_axon,mean_radi_nerve,mean_area_nerve,mean_radi_myelin_thickness,mean_area_myelin_thickness



def nerve_density_count():
    """
    Manual (tenemos que contar el numero de nervios que tenemos en una foto)ç
    Se puede hacer con analayis blob pero es más complicado 
    """

    pass



