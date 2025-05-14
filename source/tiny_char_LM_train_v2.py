from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np

# --- Data prep ---
text = "Hello, World! How are you? I am fine, thank you. Summer is on the way."
chars = sorted(set(text))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}
encoded = [char2idx[c] for c in text]

nr_features = 4  # smaller window
X, y = [], []
for i in range(len(encoded) - nr_features):
    X.append(encoded[i : i + nr_features])
    y.append(encoded[i + nr_features])
X = np.array(X)
y = to_categorical(y, num_classes=len(chars))

# --- Model: much simpler ---
model = Sequential(
    [
        Input(shape=(nr_features,)),
        Embedding(input_dim=len(chars), output_dim=16, input_length=nr_features),
        LSTM(32),  # single layer
        Dense(len(chars), activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# --- Train: force memorization ---
history = model.fit(X, y, epochs=500, batch_size=2, verbose=2)

# --- Evaluate & test prediction ---
loss, acc = model.evaluate(X, y, verbose=0)
print(f"Final train accuracy: {acc:.4f}, loss: {loss:.4f}")

# Try a sample prediction
i = 21
sample = X[i].reshape(1, nr_features)
pred = model.predict(sample, verbose=0)
pred_char = idx2char[np.argmax(pred)]
print(
    "Input:",
    "".join(idx2char[j] for j in X[i]),
    "→ Predicted:",
    pred_char,
    " (Expected:",
    idx2char[encoded[i + nr_features]] + ")",
)
