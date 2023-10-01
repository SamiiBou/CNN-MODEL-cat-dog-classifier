from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# Chemin vers les dossiers
TRAINING_PATH = './training_set'
TEST_PATH = './test_set'

# Initialisation des générateurs d'images
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

print("Préparation du générateur d'images pour l'entraînement :")
print("1. Chargement des images à partir du disque.")
print("2. Mise à l'échelle des pixels entre 0 et 1.")
print("3. Redimensionnement des images à une taille de 150x150.")
print("4. Regroupement des images en lots de 32.")
print("5. Étiquetage automatique basé sur les noms de sous-répertoires (cats ou dogs).")

train_generator = train_datagen.flow_from_directory(
    TRAINING_PATH,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

print("\nPréparation du générateur d'images pour le test :")
print("1. Chargement des images à partir du disque.")
print("2. Mise à l'échelle des pixels entre 0 et 1.")
print("3. Redimensionnement des images à une taille de 150x150.")
print("4. Regroupement des images en lots de 32.")
print("5. Étiquetage automatique basé sur les noms de sous-répertoires (cats ou dogs).")

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
