# Modèle CNN pour la classification Chats vs Chiens

Ce projet consiste à entraîner un modèle CNN pour classer des images en deux catégories : chats et chiens. Le modèle a été entraîné à partir d'un ensemble de données contenant des images de chats et de chiens.

## Comment utiliser ce projet

1. **Entraîner un nouveau modèle** :  
Pour entraîner un nouveau modèle, vous pouvez exécuter les trois premiers fichiers .py dans l'ordre suivant :
   
   - `data_preprocessing.py` : Préparez et traitez vos données. Assurez-vous que vos images soient dans les dossiers `./training_set` pour l'entraînement et `./test_set` pour les tests.
   - `cnn_model.py` : Ce fichier contient les fonctions pour créer et entraîner le modèle CNN.
   - `main.py` : Exécutez ce fichier pour commencer l'entraînement du modèle à partir des données prétraitées.

2. **Utiliser un modèle pré-entraîné** :  
Si vous ne souhaitez pas entraîner le modèle à partir de zéro, vous pouvez utiliser le modèle pré-entraîné fourni dans `cat_dog_model.h5`. Ce modèle a une précision d'environ 78%, ce qui signifie qu'il peut parfois se tromper.

   Pour utiliser le modèle pré-entraîné, exécutez le fichier `prediction.py`. Ce fichier ouvrira une boîte de dialogue vous permettant de sélectionner une image pour laquelle vous souhaitez obtenir une prédiction.

3. **Élargir l'ensemble de données d'entraînement** :  
Si vous avez plus d'images de chats ou de chiens et que vous souhaitez élargir votre ensemble de données d'entraînement, placez simplement ces images dans les dossiers appropriés (`./training_set/cats` pour les chats et `./training_set/dogs` pour les chiens).

## Remarques importantes :
- Assurez-vous d'avoir TensorFlow installé pour exécuter les scripts.
- Le modèle utilise une taille d'image de 150x150 pour l'entraînement et la prédiction. Assurez-vous que vos images sont d'une taille appropriée ou qu'elles peuvent être redimensionnées sans perte d'information importante.
