import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Spécifier le chemin du dossier 
folder_path = 'C:\\Users\\louhe\\OneDrive\\Bureau\\Stage\\mammo_with_bacs_annotations\\masks'
# Obtenir la liste de tous les fichiers .nii dans le dossier
nii_files = [f for f in os.listdir(folder_path) if f.endswith('.nii')]

for nii_file in nii_files:
    # Charger le fichier .nii
    img = nib.load(os.path.join(folder_path, nii_file))
    data = img.get_fdata()
    
    # Pour cet exemple, nous allons convertir seulement la coupe médiane de l'image 3D en .png
    middle_slice = data[:, :, data.shape[2] // 2]
    
    # Transposer la coupe médiane
    middle_slice = middle_slice.T
    
    # Sauvegarder la coupe en .png
    plt.imsave(os.path.join(folder_path, nii_file.replace('.nii', '.png')), middle_slice, cmap='gray')

print("Conversion terminée !")
