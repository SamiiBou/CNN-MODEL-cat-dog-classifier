from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog

model_path = "cat_dog_model.h5"  # Remplacez par votre chemin si nécessaire.
model = load_model(model_path)

def preprocess_image(image_path):
    # Charger l'image
    img = image.load_img(image_path, target_size=(150, 150))
    # Convertir en tableau numpy et rescaler
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    
    return img_array

def predict_image_class(model, img_array):
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        return "Dog", prediction[0][0]
    else:
        return "Cat", 1 - prediction[0][0]

def predict_from_image_path(model):
    root = tk.Tk()  # Créer une fenêtre TK root (il faut qu'elle soit créée puis cachée)
    root.withdraw()  # Cacher la fenêtre TK
    
    # Ouvrir l'explorateur de fichiers et obtenir le chemin du fichier sélectionné
    image_path = filedialog.askopenfilename()
    
    if not image_path:
        print("Aucun fichier sélectionné.")
        return

    try:
        img_array = preprocess_image(image_path)
        class_name, confidence = predict_image_class(model, img_array)
    
        print(f"La prédiction pour l'image {image_path} est: {class_name} (Confidence: {confidence*100:.2f}%)")
    except Exception as e:
        print(f"Erreur lors de la lecture de l'image. Assurez-vous que le chemin est correct et que l'image est valide. Erreur: {e}")

predict_from_image_path(model)
