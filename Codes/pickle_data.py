import os
import pickle
import matplotlib.pyplot as plt

# Remplacez "path_to_your_pkl_files" par le chemin d'accès réel à votre dossier contenant les fichiers .pkl
path_t = "C://Users//louhe//OneDrive//Bureau//Stâge//mammo_with_bacs_annotations//dicoms_2//test_pro"

# Liste tous les fichiers dans le dossier
files = os.listdir(path_t)

# Boucle sur tous les fichiers
for file in files:
    # Vérifie si le fichier est un fichier .pkl
    if file.endswith(".pkl"):
        # Ouvre le fichier .pkl
        with open(os.path.join(path_t, file), "rb") as f:
            img = pickle.load(f)

        # Affiche l'image
        plt.imshow(img[0], cmap='gray')
        plt.show()
