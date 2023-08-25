import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize
import time

def find_endpoints(img):

    height, width = img.shape
    result = np.zeros((height, width), np.uint8)

    for y in range(1, height - 1):
        for x in range(1, width - 1):

            if img[y, x] > 0:

                neighborhood = img[y - 1:y + 2, x - 1:x + 2]
                non_zero_count = np.count_nonzero(neighborhood)

                if non_zero_count < 3:

                    result[y, x] = 255

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
        new_direction = next_point - current_point
        current_direction = 0.3 * current_direction + 0.7 * new_direction
        current_direction /= np.linalg.norm(current_direction)
        #current_direction = next_point - current_point
    return path

#def main(vesselness_path, bac_path, original_image_path, patch_size=500):
    start_time = time.time()
    bac_image = cv2.imread(bac_path, cv2.IMREAD_GRAYSCALE)
    vesselness = cv2.imread(vesselness_path, cv2.IMREAD_GRAYSCALE)
    original_image = plt.imread(original_image_path)

    target_size = original_image.shape
    #print(bac_image.shape)
    target_size = (target_size[1], target_size[0])
    vesselness = cv2.resize(vesselness, target_size)
    #print(vesselness.shape)

    skeleton = skeletonize(bac_image > 0)
    end_points = find_endpoints(skeleton)

    # Combine bac_image et the vesselness
    vesselness = np.where(vesselness > 100, 255, 0) #pour 100 pas mal du tout
    combined_image = np.maximum(bac_image, vesselness)

    #plt.figure()
    #plt.imshow(skeleton)
    #plt.show()

    plt.figure()
    for i in range(0, bac_image.shape[0], patch_size):
        for j in range(0, bac_image.shape[1], patch_size):
            bac_patch = skeleton[i:i+patch_size, j:j+patch_size]
            seed_patch = end_points[i:i+patch_size, j:j+patch_size]
            combined_patch = combined_image[i:i+patch_size, j:j+patch_size]
            
            paths = [follow_path(seed, combined_patch, bac_patch, search_window_size=7, neighborhood_size=10) for seed in np.argwhere(seed_patch > 0)]
        
            for path in paths:
                plt.plot([j + point[1] for point in path], [i + point[0] for point in path], color='r')

    end_time = time.time()
    print(f"Running time: {end_time - start_time} seconds")
    #end_points_img = np.argwhere(end_points > 0)
    #plt.scatter(end_points_img[:, 1], end_points_img[:, 0], color='b', s=50)
    #plt.imshow(original_image, cmap='gray')
    plt.imshow(bac_image, cmap= 'Reds')#, alpha=0.5, cmap='Reds')
    plt.savefig(output_image_path)
    plt.show()

def main(vesselness_path, bac_path, patch_size=200):

    start_time = time.time()
    bac_image = cv2.imread(bac_path, cv2.IMREAD_GRAYSCALE)
    vesselness = cv2.imread(vesselness_path, cv2.IMREAD_GRAYSCALE)

    # Créer une copie de bac_image pour dessiner les chemins en blanc
    bac_image_copy = np.copy(bac_image)

    target_size = bac_image.shape
    #print(bac_image.shape)
    target_size = (target_size[1], target_size[0])
    vesselness = cv2.resize(vesselness, target_size)
    #print(vesselness.shape)

    skeleton = skeletonize(bac_image > 0)
    end_points = find_endpoints(skeleton)

    # Combine bac_image et the vesselness
    vesselness = np.where(vesselness > 100, 255, 0) #pour 100 pas mal du tout
    combined_image = np.maximum(bac_image, vesselness)

    #plt.figure()
    #plt.imshow(skeleton)
    #plt.show()

    plt.figure()

    for i in range(0, bac_image.shape[0], patch_size):
        for j in range(0, bac_image.shape[1], patch_size):

            bac_patch = skeleton[i:i+patch_size, j:j+patch_size]
            seed_patch = end_points[i:i+patch_size, j:j+patch_size]
            combined_patch = combined_image[i:i+patch_size, j:j+patch_size]
            
            paths = [follow_path(seed, combined_patch, bac_patch, search_window_size=15, neighborhood_size=30) for seed in np.argwhere(seed_patch > 0)]
        
            for path in paths:
                plt.plot([j + point[1] for point in path], [i + point[0] for point in path], color='r')

                for k in range(len(path) - 1):
                    start_point = (j + path[k][1], i + path[k][0])
                    end_point = (j + path[k + 1][1], i + path[k + 1][0])
                    cv2.line(bac_image_copy, start_point, end_point, 255, 1)  # Dessiner en blanc sur la copie

    end_time = time.time()
    print(f"Running time: {end_time - start_time} seconds")

    # Enregistrer la copie en noir et blanc
    cv2.imwrite(output_image_path, bac_image_copy)

    plt.imshow(bac_image, cmap='Reds')
    plt.show()


if __name__ == "__main__":

    vesselness_path = "C:/Users/louhe/RF-UNet/save_picture_NEW/EEB00A05.png"
    bac_path = "C:/Users/louhe/OneDrive/Bureau/Stage/poubelle1/EEB00A05whole/EEB00A05_premask_0.65.png"
    output_image_path = "C:/Users/louhe/OneDrive/Bureau/Stage/RESULTATS/output_image_EEB00A05_2.png"

    main(vesselness_path, bac_path)
