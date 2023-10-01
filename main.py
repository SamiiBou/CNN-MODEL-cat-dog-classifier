from data_preprocessing import TRAINING_PATH, TEST_PATH, train_datagen, test_datagen, train_generator, test_generator
from cnn_model import create_cnn_model, train_cnn_model, evaluate_cnn_model

# Créer le modèle CNN
input_shape = (150, 150, 3)  # 3 pour les canaux RGB
model = create_cnn_model(input_shape)

# Entraîner le modèle CNN
train_cnn_model(model, train_generator, test_generator)

# Evaluer le modèle
evaluate_cnn_model(model, test_generator)