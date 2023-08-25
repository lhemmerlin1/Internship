import cv2
import numpy as np


def calculate_iou(image1_path, image2_path):

    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # If not binary images
    #_, image1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    #_, image2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

    # Calculer l'intersection et l'union
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    
    iou_score = np.sum(intersection) / np.sum(union) * 100

    return iou_score

def main():
    nifti_path = "C:\\Users\\louhe\\OneDrive\\Bureau\\Stage\\mammo_with_bacs_annotations\\masks\\EEB00A05.png"
    image_path = "C:\\Users\\louhe\\OneDrive\\Bureau\\Stage\\poubelle1\\test_EEB00A05.png" #EE632566whole\\EE632566_premask_0.65.png"
    #image_path = "C:\\Users\\louhe\\OneDrive\\Bureau\\Stage\\RESULTATS\\erase\\erase_EEB00A05.png"
    #image_path = "C:\\Users\\louhe\\OneDrive\\Bureau\\Stage\\RESULTATS\\output_image_EEB00A05.png"
    iou_score = calculate_iou(nifti_path, image_path)

    if iou_score is not None:
        print(f"Jaccard's coeff is {iou_score:.2f}%")

if __name__ == "__main__":
    main()
