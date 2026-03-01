from tensorflow.keras import layers, Model

WINDOW = 256
CHANNELS = 8
EMBED_DIM = 128

def build_encoder():
    inp = layers.Input(shape=(WINDOW, CHANNELS))

    # Feature Extraction
    x = layers.Conv1D(64, 5, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 5, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, 3, activation="relu")(x)
    
    # Reverting to GAP: Faster, more stable for small datasets
    x = layers.GlobalAveragePooling1D()(x)

    # Projection Head
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(EMBED_DIM)(x)

    # Crucial Hypersphere Normalization
    x = layers.UnitNormalization(axis=-1)(x)

    return Model(inp, x)
