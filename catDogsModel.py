from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Chemin vers les dossiers
TRAINING_PATH = './training_set'
TEST_PATH = './test_set'

# Initialisation des générateurs d'images
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    TRAINING_PATH,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
