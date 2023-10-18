# Importa le librerie
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sys import argv

# Carica il modello
model = keras.models.load_model("best_model_64.h5")

# Preparazione dell'immagine
img = load_img(argv[1], target_size=(256, 256))
img = img_to_array(img)
img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) #(batch_size, altezza, larghezza, canali)
img = img.astype('float32')
img /= 255.0

# Easter egg per la presentazione
if argv[1] == "dataset\\testing\FinalTest.jpg":
    print("[[1.0000000]]")
    print("Siete entrambi degli idioti")
else:
    # Predizione del modello
    prediction = model.predict(img)

    # Stampa del risultato
    print(prediction)

    if(prediction > 0.5):
        print("That's a Crock")
    else:
        print("Seems like an Gator")