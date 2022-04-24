import numpy as np
from typing import Tuple
from skimage.draw import polygon2mask
import scipy.interpolate
from visualize_data import visualize



def create_circle(center: Tuple[float, float], r: float, n: int) -> np.ndarray:

    alpha = np.linspace(0, 2 * np.pi, n)[:-1]


    return np.array([center[0] + r * np.cos(alpha), center[1] + r * np.sin(alpha)]).T

def create_multiple_circles(centers: Tuple[float, float], rs: float, n: int) -> np.ndarray:

    alpha = np.linspace(0, 2 * np.pi, n)[:-1]


    return [np.array([center[0] + r * np.cos(alpha), center[1] + r * np.sin(alpha)]).T for center, r in zip(centers, rs)]


def get_snake_normals(snake):
    normals = []
    for i in range(len(snake)):
        if i == 0:
            dx = snake[-1, 0] - snake[i+1, 0]
            dy = snake[-1, 1] - snake[i+1, 1]

            normal = np.array([-dy, dx])

            normal = normal / np.linalg.norm(normal)

            normals.append(normal)
        elif i == len(snake)-1:
            dx = snake[i-1, 0] - snake[0, 0]
            dy = snake[i-1, 1] - snake[0, 1]

            normal = np.array([-dy, dx])

            normal = normal / np.linalg.norm(normal)

            normals.append(normal)
        else:
            dx = snake[i-1, 0] - snake[i+1, 0]
            dy = snake[i-1, 1] - snake[i+1, 1]

            normal = np.array([-dy, dx])

            normal = normal / np.linalg.norm(normal)

            normals.append(normal)
    
    return np.array(normals)


def curve_smoothing_smooth(a, b, X):
    n = len(X)

    A = np.diag(np.full(n, -2)) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1) + np.diag(np.ones(1), n-1) + np.diag(np.ones(1), -n+1)
    B = np.diag(np.full(n, -6)) + np.diag(np.full(n-1, 4), 1) + np.diag(np.full(n-1, 4), -1) + np.diag(np.full(1, 4), n-1) + np.diag(np.full(1, 4), -n+1) + \
        np.diag(np.full(n-2, -1), 2) + np.diag(np.full(n-2, -1), -2) + np.diag(np.full(2, -1), n-2) + np.diag(np.full(2, -1), -n+2) 

    return np.dot(np.linalg.inv(np.identity(n) - a * A - b * B), X)


def distribute_points(snake):
    """ Distributes snake points equidistantly."""
    N = snake.shape[1]
    d = np.sqrt(np.sum((np.roll(snake, -1, axis=1)-snake)**2, axis=0)) # length of line segments
    f = scipy.interpolate.interp1d(np.r_[0, np.cumsum(d)], np.c_[snake, snake[:,0:1]])
    return(f(sum(d)*np.arange(N)/N))

def is_crossing(p1, p2, p3, p4):
    """ Check if the line segments (p1, p2) and (p3, p4) cross."""
    crossing = False
    d21 = p2 - p1
    d43 = p4 - p3
    d31 = p3 - p1
    det = d21[0]*d43[1] - d21[1]*d43[0] # Determinant
    if det != 0.0 and d21[0] != 0.0 and d21[1] != 0.0:
        a = d43[0]/d21[0] - d43[1]/d21[1]
        b = d31[1]/d21[1] - d31[0]/d21[0]
        if a != 0.0:
            u = b/a
            if d21[0] > 0:
                t = (d43[0]*u + d31[0])/d21[0]
            else:
                t = (d43[1]*u + d31[1])/d21[1]
            crossing = 0 < u < 1 and 0 < t < 1         
    return crossing

def is_counterclockwise(snake):
    """ Check if points are ordered counterclockwise."""
    return np.dot(snake[0,1:] - snake[0,:-1],
                  snake[1,1:] + snake[1,:-1]) < 0

def remove_intersections(snake):
    """ Reorder snake points to remove self-intersections.
        Arguments: snake represented by a 2-by-N array.
        Returns: snake.
    """
    pad_snake = np.append(snake, snake[:,0].reshape(2,1), axis=1)
    pad_n = pad_snake.shape[1]
    n = pad_n - 1 
    
    for i in range(pad_n - 3):
        for j in range(i + 2, pad_n - 1):
            pts = pad_snake[:,[i, i + 1, j, j + 1]]
            if is_crossing(pts[:,0], pts[:,1], pts[:,2], pts[:,3]):
                # Reverse vertices of smallest loop
                rb = i + 1 # Reverse begin
                re = j     # Reverse end
                if j - i > n // 2:
                    # Other loop is smallest
                    rb = j + 1
                    re = i + n                    
                while rb < re:
                    ia = rb % n
                    rb = rb + 1                    
                    ib = re % n
                    re = re - 1                    
                    pad_snake[:,[ia, ib]] = pad_snake[:,[ib, ia]]                    
                pad_snake[:,-1] = pad_snake[:,0]                
    snake = pad_snake[:,:-1]
    if is_counterclockwise(snake):
        return snake
    else:
        return np.flip(snake, axis=1)

def constraint_snake(img, snake):
    snake[snake < 0] = 0
    snake[:, 0][snake[:, 0] > img.shape[0]] = img.shape[0]
    snake[:, 1][snake[:, 1] > img.shape[1]] = img.shape[1]

    return snake


def calculate_intensities(img, snake, offside_out, offside_in):
    snake_out = snake + get_snake_normals(snake) * offside_out
    snake_in = snake + get_snake_normals(snake) * offside_in

    intensity_out = img[np.round(snake_out).astype(int)].mean()
    intensity_in = img[np.round(snake_in).astype(int)].mean()

    return intensity_out, intensity_in


def deformable(img, snake, tau=2, alpha=0.2, beta=0.2, offside_out = 1):
    
    # mean_outside, mean_inside = calculate_intensities(img, snake, 1, -1)
    # print(mean_outside, mean_inside)
    mean_outside, mean_inside = snake_mask(img, snake, offside_out)
    # print(mean_outside, mean_inside)


    F_ext = []
    for point in snake:
        point = np.round(point).astype(int)
        I = img[point[0], point[1]]
        f_ext = (mean_inside - mean_outside) * (2*I - mean_inside - mean_outside)
        F_ext.append(f_ext)

    F_ext = np.array(F_ext)

    idx = np.round(snake).astype(int)
    I = img[idx[:,0], idx[:,1]]
    
    F_ext = (mean_inside - mean_outside) * (2*I - mean_inside - mean_outside)


    # print(mean_inside, mean_outside)

    snake_normals = get_snake_normals(snake)

    snake = curve_smoothing_smooth(alpha, beta, snake + tau * F_ext[:, None] * snake_normals)
    
    snake = constraint_snake(img, snake)
    snake = distribute_points(snake.T)
    snake = remove_intersections(snake).T

    return snake

def snake_mask(img, snake, offside_out):

    snake_out = snake + get_snake_normals(snake) * offside_out

    mask = polygon2mask(img.shape, snake)
    mask_out = polygon2mask(img.shape, snake_out)
    mask_out = mask_out & ~mask

    intensity_in = np.mean(img[mask])

    intensity_out = np.mean(img[mask_out])

    return intensity_out, intensity_in
