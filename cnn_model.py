from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os


def create_cnn_model(input_shape):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_cnn_model(model, train_generator, test_generator, epochs=25):
    model_path = os.path.join(os.path.dirname(__file__), 'cat_dog_model.h5')
    
    # Afficher le chemin et demander la confirmation
    print(f"Le modèle sera sauvegardé à : {model_path}")
    choice = input("Voulez-vous continuer et entraîner le modèle? (y/n): ")
    
    if choice.lower() != 'y':
        print("Entraînement annulé.")
        return model
    
    # Entraînement du modèle
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
    )
    
    model.save(model_path)
    print("Modèle sauvegardé avec succès.")
        
    return model

def evaluate_cnn_model(model, test_generator):
    """Évaluer le modèle sur l'ensemble de test."""
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy
