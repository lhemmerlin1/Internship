import os
import pydicom
from PIL import Image
import numpy as np

#input_folder = "C:\\Users\\louhe\\Downloads\\DICOM"
#output_folder = "C:\\Users\\louhe\\Downloads\\DICOM_2"
input_folder = "C:\\Users\\louhe\\OneDrive\\Bureau\\Stage\\mammo_with_bacs_annotations\\dicoms"
output_folder = "C:\\Users\\louhe\\OneDrive\\Bureau\\Stage\\mammo_with_bacs_annotations\\dicoms_4"

# Crée le répertoire de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.dcm'):
        print(f"Traitement du fichier {filename}")  # Affiche le nom du fichier en cours de traitement
        try:
            # Charge le fichier DICOM
            path = os.path.join(input_folder, filename)
            ds = pydicom.dcmread(path)

            # Convertit l'image DICOM en PIL Image
            im = Image.fromarray(ds.pixel_array)

            # Convertit l'image en 8 bits
            image_16bit = np.array(im)
            image_8bit = ((image_16bit - image_16bit.min()) / (image_16bit.ptp() / 255.0)).astype(np.uint8)
            image_8bit = image_8bit  # 255 - image_8bit to Inverse l'échelle de gris
            im_8bit = Image.fromarray(image_8bit)

            # Enregistre l'image au format JPG
            new_filename = os.path.splitext(filename)[0] + '.png'  # Ajoute l'extension '.jpg' au nom de fichier
            im_8bit.save(os.path.join(output_folder, new_filename), "PNG")
            print(f"Image sauvegardée : {new_filename}")  # Affiche le nom de l'image sauvegardée
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {filename}: {e}")  # Affiche toute erreur survenue
    else:
        print(f"Le fichier {filename} n'est pas un fichier .dcm")  # Affiche un message si le fichier n'est pas un fichier .dcm
