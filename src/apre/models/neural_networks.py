
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def create_autoencoder_pipeline(X_scaled, encoding_dim=32, epochs=50):
    print("Training Autoencoder...")
    input_dim = X_scaled.shape[1]
    
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    # Simple architecture
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    encoded = layers.Dense(encoding_dim, activation='relu', name='encoded')(x)
    
    # Decoder
    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.Dense(128, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='linear')(x)
    
    autoencoder = models.Model(inputs, decoded)
    encoder = models.Model(inputs, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    print("Fitting Autoencoder...")
    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=32,
        shuffle=True,
        verbose=0
    )
    
    X_encoded = encoder.predict(X_scaled)
    return X_encoded

def build_dec_model(input_dim, n_clusters):
    # Simplified DEC model
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    z = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(n_clusters, activation='softmax')(z)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

def train_dec_model(model, X, cluster_labels, epochs=30):
    print("Training specialized DEC model (supervised by initial clusters)...")
    # Using initial cluster labels as pseudo-labels
    model.fit(X, cluster_labels, epochs=epochs, batch_size=32, verbose=0)
    proba = model.predict(X)
    return proba

def create_cnn_pipeline(X_encoded, y_encoded, test_size=0.2, epochs=30):
    print("Training CNN on encoded features...")
    
    # Reshape for 1D CNN: (samples, features, 1)
    X_reshaped = X_encoded.reshape((X_encoded.shape[0], X_encoded.shape[1], 1))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    n_classes = len(np.unique(y_encoded))
    
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=(X_encoded.shape[1], 1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    return results
