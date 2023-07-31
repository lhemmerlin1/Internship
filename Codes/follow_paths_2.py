import numpy as np
import cv2
import matplotlib.pyplot as plt
from pyrsistent import b
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize
import time
from scipy.ndimage import convolve

import cv2
import numpy as np

def find_endpoints(img):
    # Obtention des dimensions de l'image
    height, width = img.shape

    # Créer une nouvelle image noire pour le résultat
    result = np.zeros((height, width), np.uint8)

    # Itérer à travers chaque pixel de l'image
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Si le pixel n'est pas noir (c'est-à-dire s'il fait partie d'un segment)
            if img[y, x] > 0:
                # Extraire le voisinage 3x3
                neighborhood = img[y - 1:y + 2, x - 1:x + 2]
                
                # Compter le nombre de pixels non nuls dans le voisinage
                non_zero_count = np.count_nonzero(neighborhood)
                
                # Si le pixel a moins de 3 voisins non nuls, il est considéré comme un point d'extrémité ou de branchement
                if non_zero_count < 3:
                    result[y, x] = 255  # Peindre le pixel en blanc

    return result

def get_candidate_points(current_point, vesselness, search_window_size):
    search_window = vesselness[max(0, current_point[0] - search_window_size):current_point[0] + search_window_size + 1,
                               max(0, current_point[1] - search_window_size):current_point[1] + search_window_size + 1]
    candidate_points = np.argwhere(search_window > 0)
    candidate_points += [max(0, current_point[0] - search_window_size), max(0, current_point[1] - search_window_size)]
    return candidate_points

def get_initial_direction(seed, vesselness, neighborhood_size):
    neighborhood = vesselness[max(0, seed[0] - neighborhood_size):seed[0] + neighborhood_size + 1,
                              max(0, seed[1] - neighborhood_size):seed[1] + neighborhood_size + 1]
    
    points = np.argwhere(neighborhood > 0)
    if points.size == 0 or points.shape[0] < 2:
        return np.array([1, 0])
    pca = PCA(n_components=2)
    pca.fit(points)
    initial_direction = pca.components_[np.argmax(pca.explained_variance_)]
    return initial_direction

def choose_next_point(current_point, candidate_points, vesselness, current_direction, visited_points):
    candidate_points = np.array([point for point in candidate_points if tuple(point) not in visited_points])
    if candidate_points.size == 0:
        return None
    directions_to_candidates = candidate_points - current_point
    cosine_similarity = np.dot(directions_to_candidates, current_direction) / (np.linalg.norm(directions_to_candidates, axis=1) * np.linalg.norm(current_direction))
    cosine_similarity = np.clip(cosine_similarity, -1, 1)
    angles = np.arccos(cosine_similarity)
    valid_candidates = candidate_points[angles < np.pi]
    if valid_candidates.size == 0:
        return None
    next_point = valid_candidates[np.argmax(vesselness[valid_candidates[:, 0], valid_candidates[:, 1]])]
    return next_point

def follow_path(seed, vesselness, bac_image, search_window_size, neighborhood_size):
    path = [seed]
    current_direction = get_initial_direction(seed, bac_image, neighborhood_size)
    visited_points = set()
    while True:
        current_point = path[-1]
        visited_points.add(tuple(current_point))
        candidate_points = get_candidate_points(current_point, vesselness, search_window_size)
        next_point = choose_next_point(current_point, candidate_points, vesselness, current_direction, visited_points)
        if next_point is None:
            break
        path.append(next_point)
        # new_direction = next_point - current_point
        # current_direction = 0.7 * current_direction + 0.3 * new_direction
        # current_direction /= np.linalg.norm(current_direction)
        current_direction = next_point - current_point
    return path

# Here is a modified version of your code that first tries to follow the path on the bac_image and then on the vesselness_image if it gets stuck.

#def follow_path(seed, vesselness, bac_image, search_window_size, neighborhood_size):
    # path = [seed]
    # current_direction = get_initial_direction(seed, bac_image, neighborhood_size)
    # visited_points = set()
    # while True:
    #     current_point = path[-1]
    #     visited_points.add(tuple(current_point))
    #     candidate_points = get_candidate_points(current_point, bac_image, search_window_size)
    #     next_point = choose_next_point(current_point, candidate_points, bac_image, current_direction, visited_points)
    #     if next_point is None:
    #         # If we can't find a next point on the bac_image, try the vesselness_image
    #         candidate_points = get_candidate_points(current_point, vesselness, search_window_size)
    #         next_point = choose_next_point(current_point, candidate_points, vesselness, current_direction, visited_points)
    #         if next_point is None:
    #             break
    #     path.append(next_point)
    #     # new_direction = next_point - current_point
    #     # current_direction = 0.7 * current_direction + 0.3 * new_direction
    #     # current_direction /= np.linalg.norm(current_direction)
    #     current_direction = next_point - current_point
    # return path

# The rest of your code remains the same.

#def main(vesselness_path, bac_path, original_image_path, patch_size=500):
    start_time = time.time()
    bac_image = cv2.imread(bac_path, cv2.IMREAD_GRAYSCALE)
    vesselness = cv2.imread(vesselness_path, cv2.IMREAD_GRAYSCALE)
    original_image = plt.imread(original_image_path)

    target_size = bac_image.shape
    print(bac_image.shape)
    target_size = (target_size[1], target_size[0])
    vesselness = cv2.resize(vesselness, target_size)
    print(vesselness.shape)

    skeleton = skeletonize(bac_image > 0)
    end_points = find_endpoints(skeleton)

    # Combine the bac_image and the vesselness map into a single image
    vesselness = np.where(vesselness > 10, 255, 0)
    combined_image = np.maximum(bac_image, vesselness)

    # plt.figure()
    # plt.imshow(combined_image)
    # plt.show()

    plt.figure()
    for i in range(0, bac_image.shape[0], patch_size):
        for j in range(0, bac_image.shape[1], patch_size):
            bac_patch = end_points[i:i+patch_size, j:j+patch_size]
            combined_patch = combined_image[i:i+patch_size, j:j+patch_size]
            
            paths = [follow_path(seed, combined_patch, bac_image, search_window_size=5, neighborhood_size=5) for seed in np.argwhere(bac_patch > 0)]
        
            for path in paths:
                plt.plot([j + point[1] for point in path], [i + point[0] for point in path], color='r')

    end_time = time.time()
    print(f"Running time: {end_time - start_time} seconds")
    end_points_img = np.argwhere(end_points > 0)
    plt.scatter(end_points_img[:, 1], end_points_img[:, 0], color='b', s=50)
    plt.imshow(original_image, cmap='gray')
    plt.show()


# Sure, you can use the `plt.arrow` function to draw an arrow representing the initial direction vector. 
# You need to draw the arrow at the seed point with the direction given by the PCA. Here's how you can modify the main function:

def main(vesselness_path, bac_path, original_image_path, patch_size=500):
    start_time = time.time()
    bac_image = cv2.imread(bac_path, cv2.IMREAD_GRAYSCALE)
    vesselness = cv2.imread(vesselness_path, cv2.IMREAD_GRAYSCALE)
    original_image = plt.imread(original_image_path)

    target_size = bac_image.shape
    #print(bac_image.shape)
    target_size = (target_size[1], target_size[0])
    vesselness = cv2.resize(vesselness, target_size)
    #print(vesselness.shape)

    skeleton = skeletonize(bac_image > 0)
    end_points = find_endpoints(skeleton)

    # Combine the bac_image and the vesselness map into a single image
    vesselness = np.where(vesselness > 100, 255, 0) #pour 100 pas mal du tout
    combined_image = np.maximum(bac_image, vesselness)

    plt.figure()
    plt.imshow(combined_image)
    plt.show()

    plt.figure()
    for i in range(0, bac_image.shape[0], patch_size):
        for j in range(0, bac_image.shape[1], patch_size):
            bac_patch = skeleton[i:i+patch_size, j:j+patch_size]
            seed_patch = end_points[i:i+patch_size, j:j+patch_size]
            combined_patch = combined_image[i:i+patch_size, j:j+patch_size]
            
            paths = [follow_path(seed, combined_patch, bac_patch, search_window_size=10, neighborhood_size=10) for seed in np.argwhere(seed_patch > 0)]
        
            for path in paths:
                plt.plot([j + point[1] for point in path], [i + point[0] for point in path], color='r')
                # Draw the initial direction vector as an arrow
                #initial_direction = get_initial_direction(path[0], combined_patch, neighborhood_size=10)
                #plt.arrow(j + path[0][1], i + path[0][0], initial_direction[1]*50, initial_direction[0]*50, color='g', head_width=20)


    end_time = time.time()
    print(f"Running time: {end_time - start_time} seconds")
    end_points_img = np.argwhere(end_points > 0)
    plt.scatter(end_points_img[:, 1], end_points_img[:, 0], color='b', s=50)
    plt.imshow(original_image, cmap='gray')
    plt.show()


if __name__ == "__main__":
    vesselness_path = "C:/Users/louhe/RF-UNet/save_picture_NEW/pre5.png"
    bac_path = "C:/Users/louhe/OneDrive/Bureau/Stage/poubelle/EEC96467whole/EEC96467_premask_0.65.png"
    original_image_path = "C:/Users/louhe/OneDrive/Bureau/Stage/mammo_with_bacs_annotations/dicoms_2/EEC96467.jpg"
    # vesselness_path = "C:/Users/louhe/RF-UNet/save_picture_NEW/pre1.png"
    # bac_path = "C:/Users/louhe/OneDrive/Bureau/Stage/poubelle/EE61BA1Dwhole/EE61BA1D_premask_0.65.png"
    # original_image_path = "C:/Users/louhe/OneDrive/Bureau/Stage/mammo_with_bacs_annotations/dicoms_2/EE61BA1D.jpg"
    main(vesselness_path, bac_path, original_image_path)

#def main(vesselness_path, bac_path, original_image_path, patch_size=100):
    # start_time = time.time()
    # bac_image = cv2.imread(bac_path, cv2.IMREAD_GRAYSCALE)
    # vesselness = cv2.imread(vesselness_path, cv2.IMREAD_GRAYSCALE)
    # original_image = plt.imread(original_image_path)

    # target_size = bac_image.shape
    # print(bac_image.shape)
    # target_size = (target_size[1], target_size[0])
    # vesselness = cv2.resize(vesselness, target_size)
    # print(vesselness.shape)

    # skeleton = skeletonize(bac_image>0)
    # end_points = find_endpoints(skeleton)
    # # plt.figure()
    # # plt.imshow(end_points)
    # # plt.show()
    # plt.figure()
    # for i in range(0, bac_image.shape[0], patch_size):
    #     for j in range(0, bac_image.shape[1], patch_size):
    #         bac_patch = end_points[i:i+patch_size, j:j+patch_size]
    #         vesselness_patch = vesselness[i:i+patch_size, j:j+patch_size]
            
    #         paths = [follow_path(seed, vesselness_patch, search_window_size=5, neighborhood_size=10) for seed in np.argwhere(bac_patch > 0)]
        
    #         for path in paths:
    #             plt.plot([j + point[1] for point in path], [i + point[0] for point in path], color='r')

    # end_time = time.time()
    # print(f"Running time: {end_time - start_time} seconds")
    # end_points_img = np.argwhere(end_points > 0)
    # plt.scatter(end_points_img[:, 1], end_points_img[:, 0], color='b', s=50)
    # plt.imshow(original_image, cmap='gray')
    # plt.show()


# def main(vesselness_path, bac_path, original_image_path, patch_size=100, overlap=25):
#     start_time = time.time()
#     bac_image = cv2.imread(bac_path, cv2.IMREAD_GRAYSCALE)
#     vesselness = cv2.imread(vesselness_path, cv2.IMREAD_GRAYSCALE)
#     original_image = plt.imread(original_image_path)

#     target_size = bac_image.shape
#     target_size = (target_size[1], target_size[0])
#     vesselness = cv2.resize(vesselness, target_size)

#     plt.figure()
#     skeleton = skeletonize(bac_image>0)
#     # Divide the image into overlapping patches and process each patch separately
#     for i in range(0, bac_image.shape[0] - patch_size, patch_size - overlap):
#         for j in range(0, bac_image.shape[1] - patch_size, patch_size - overlap):
#             bac_patch = skeleton[i:i+patch_size, j:j+patch_size]
#             vesselness_patch = vesselness[i:i+patch_size, j:j+patch_size]

#             paths = [follow_path(seed, vesselness_patch, search_window_size=30, neighborhood_size=30) for seed in np.argwhere(bac_patch > 0)]

#             for path in paths:
#                 plt.plot([j + point[1] for point in path], [i + point[0] for point in path], color='r')

#     plt.imshow(original_image, cmap='gray')
#     plt.show()
#     end_time = time.time()
#     print(f"Running time: {end_time - start_time} seconds")


# import cv2
# import numpy as np
# from scipy.ndimage import convolve
# from skimage.morphology import skeletonize
# import matplotlib.pyplot as plt

# from scipy.ndimage import label
# from skimage.morphology import skeletonize

# def get_endpoints_and_discontinuities_1(bac_image):
#     skeleton = skeletonize(bac_image > 0)
#     plt.figure()
#     plt.imshow(skeleton)
#     plt.show
#     kernel1 = np.array([[1, 0, 0],
#                     [0, 0, 0],
#                     [0, 0, 0]])
    
#     kernel2 = np.array([[0, 1, 0],
#                     [0, 0, 0],
#                     [0, 0, 0]])
    
#     kernel3 = np.array([[0, 0, 1],
#                     [0, 0, 0],
#                     [0, 0, 0]])
    
#     kernel4 = np.array([[0, 0, 0],
#                     [1, 0, 0],
#                     [0, 0, 0]])
    
#     kernel5 = np.array([[0, 0, 0],
#                     [0, 1, 1],
#                     [0, 0, 0]])
    
#     kernel6 = np.array([[0, 0, 0],
#                     [0, 1, 0],
#                     [1, 0, 0]])
    
#     kernel7 = np.array([[0, 0, 0],
#                     [0, 1, 0],
#                     [0, 1, 0]])
    
#     kernel8 = np.array([[0, 0, 0],
#                     [0, 1, 0],
#                     [0, 0, 1]])

#     neighbors1 = convolve(skeleton, kernel1, mode='constant', cval=0)
#     neighbors2 = convolve(skeleton, kernel2, mode='constant', cval=0)
#     neighbors3 = convolve(skeleton, kernel3, mode='constant', cval=0)
#     neighbors4 = convolve(skeleton, kernel4, mode='constant', cval=0)
#     neighbors5 = convolve(skeleton, kernel5, mode='constant', cval=0)
#     neighbors6 = convolve(skeleton, kernel6, mode='constant', cval=0)
#     neighbors7 = convolve(skeleton, kernel7, mode='constant', cval=0)
#     neighbors8 = convolve(skeleton, kernel8, mode='constant', cval=0)
#     neighbors = neighbors1 + neighbors2+ neighbors3+ neighbors4
#     plt.figure()
#     plt.imshow(neighbors)
#     plt.show
#     endpoints = np.argwhere((skeleton > 0) & (neighbors == 1))

#     return endpoints


# def get_endpoints_and_discontinuities_2(bac_image):
#     skeleton = skeletonize(bac_image > 0)

#     kernel = np.array([[0, 1, 0],
#                     [1, 0, 1],
#                     [0, 1, 0]])


#     neighbors = convolve(skeleton, kernel, mode='constant', cval=0)
#     endpoints = np.argwhere((skeleton > 0) & (neighbors == 1))

#     return endpoints


# def main(bac_path):
#     bac_image = cv2.imread(bac_path, cv2.IMREAD_GRAYSCALE)

#     endpoints_1 = get_endpoints_and_discontinuities_1(bac_image)
#     endpoints_2 = get_endpoints_and_discontinuities_2(bac_image)

#     plt.figure()
#     plt.imshow(bac_image, cmap='gray')
#     plt.plot(endpoints_1[:, 1], endpoints_1[:, 0], 'go')
#     plt.title('Method 1')

#     plt.figure()
#     plt.imshow(bac_image, cmap='gray')
#     plt.plot(endpoints_2[:, 1], endpoints_2[:, 0], 'go')
#     #plt.plot(discontinuities_2[:, 1], discontinuities_2[:, 0], 'ro')
#     plt.title('Method 2')

#     plt.show()

# if __name__ == "__main__":
#     bac_path = 'C:/Users/louhe/OneDrive/Bureau/Stage/poubelle/EE61BA1Dwhole/EE61BA1D_premask_0.65.png'
#     main(bac_path)