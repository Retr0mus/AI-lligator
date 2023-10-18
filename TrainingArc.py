# Importo le librerie necessarie
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Creo un generatore di immagini che applica alcune trasformazioni casuali per aumentare il dataset
datagen = ImageDataGenerator(
    rescale=1./255, # Normalizzo i valori dei pixel tra 0 e 1
    rotation_range=20, # Ruoto le immagini di un angolo casuale tra -20 e 20 gradi
    width_shift_range=0.2, # Sposto le immagini orizzontalmente di una frazione casuale tra -0.2 e 0.2
    height_shift_range=0.2, # Sposto le immagini verticalmente di una frazione casuale tra -0.2 e 0.2
    horizontal_flip=True, # Capovolgo le immagini orizzontalmente con una probabilità del 50%
    validation_split=0.2, # Imposto una frazione del 20% delle immagini come dati di validazione
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Creo due generatori di flusso dalle cartelle delle immagini di training e validazione
train_generator = datagen.flow_from_directory(
    directory='dataset-serio/training', # La cartella che contiene le sottocartelle delle classi
    target_size=(256, 256), # La dimensione delle immagini da ridimensionare
    batch_size=32, # Il numero di immagini per ogni batch
    class_mode='binary', # La modalità di etichettatura delle classi (0 per coccodrillo, 1 per alligatore)
    subset='training' # Il sottoinsieme dei dati da usare come training
)

validation_generator = datagen.flow_from_directory(
    directory='dataset-serio/testing', # La stessa cartella del training
    target_size=(256, 256), # La stessa dimensione del training
    batch_size=32, # Lo stesso batch size del training
    class_mode='binary', # La stessa modalità di classe del training
    subset='validation' # Il sottoinsieme dei dati da usare come validazione
)   

# Creo il modello della CNN con due hidden layers da 200 nodi l'uno
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)), # Un layer convoluzionale con 32 filtri da 3x3 e funzione di attivazione ReLU
    layers.MaxPooling2D((2, 2)), # Un layer di pooling che riduce la dimensione delle feature map di un fattore 2
    layers.Conv2D(64, (3, 3), activation='relu'), # Un altro layer convoluzionale con 64 filtri da 3x3 e funzione di attivazione ReLU
    layers.MaxPooling2D((2, 2)), # Un altro layer di pooling che riduce la dimensione delle feature map di un fattore 2
    layers.Flatten(), # Un layer che appiattisce le feature map in un vettore unidimensionale 
    layers.Dropout(0.5), # Aggiungo uno strato di dropout con una probabilità del 50% prima dello strato completamente connesso
    layers.Dense(200, activation='relu'), # Un layer denso (fully connected) con 200 nodi e funzione di attivazione ReLU (primo hidden layer)
    layers.Dense(200, activation='relu'), # Un altro layer denso con 200 nodi e funzione di attivazione ReLU (secondo hidden layer)
    layers.Dense(1, activation='sigmoid') # Un layer denso con un solo nodo e funzione di attivazione sigmoide (output layer)
])

# Compilo il modello specificando la funzione di perdita (loss), l'ottimizzatore e la metrica da monitorare
model.compile(
    loss='binary_crossentropy', # La funzione di perdita per la classificazione binaria
    optimizer='adam', # L'ottimizzatore Adam che adatta il tasso di apprendimento in base al gradiente
    metrics=['accuracy'] # La metrica da monitorare è l'accuratezza della classificazione
)

# Creo una callback che salva il modello migliore in base alla metrica di validazione
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='best_model.h5', # Il nome del file in cui salvare il modello
    monitor='val_accuracy', # La metrica da monitorare è l'accuratezza di validazione
    save_best_only=True, # Salvo solo il modello che ha il valore migliore della metrica
    mode='max', # Il valore migliore della metrica è il massimo
    verbose=1 # Mostro un messaggio quando salvo il modello
)

# Addestro il modello usando i generatori di immagini e la callback
with tf.device('/GPU:0'):
    model.fit(
        train_generator, # Il generatore di immagini di training
        epochs=10, # Il numero di epoche da eseguire
        validation_data=validation_generator, # Il generatore di immagini di validazione
        callbacks=[checkpoint] # La lista delle callback da usare
    )