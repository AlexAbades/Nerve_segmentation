
import numpy as np
import scipy.interpolate
import scipy.linalg
import skimage.draw

def make_circular_snake(N, center, radius):
    """ Initialize circular snake as a 2-by-N array."""
    center = center.reshape([2,1])
    angles = np.arange(N)*2*np.pi/N
    return(center+radius*np.array([np.cos(angles), np.sin(angles)]))


def normalize(n):
    return n/np.sqrt(np.sum(n**2,axis=0))

def snake_normals(snake):
    """ Returns snake normals. Expects snake to be 2-by-N array."""
    #This can be done by averaging the normals of two line segments. 
    ds = normalize(np.roll(snake, 1, axis=1) - snake) 
    tangent = normalize(np.roll(ds,-1,axis=1) + ds)
    normal = tangent[[1,0],:]*np.array([-1,1]).reshape([2,1])
    return(normal)


def regularization_matrix(N, alpha, beta):
    """ Matrix for smoothing the snake Bint."""
    d = alpha*np.array([-2, 1, 0, 0]) + beta*np.array([-6, 4, -1, 0])
    D = np.fromfunction(lambda i,j: np.minimum((i-j)%N,(j-i)%N), (N,N), dtype=np.int)
    A = d[np.minimum(D,len(d)-1)]
    return(scipy.linalg.inv(np.eye(N)-A))