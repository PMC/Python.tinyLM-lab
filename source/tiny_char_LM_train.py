from icecream import ic
import os
from tensorflow.keras.models import Sequential
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Input,
    Embedding,
    LSTM,
    Bidirectional,
)
from tensorflow.keras.utils import to_categorical
import numpy as np

# set linewidth to infinity
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# ic.configureOutput(prefix="", outputFunction=lambda s: print(s.ljust(120)))

text = "Hello, World! How are you? I am fine, thank you. Summer is on the way."

# Create a dictionary with unique characters as keys and their indices as values
myset = set(text)
mysorted = sorted(myset)
mydict = dict(zip(mysorted, range(len(mysorted))))
idx2char = dict(zip(range(len(mysorted)), mysorted))

# Create a list of indices corresponding to the characters in the text
encoded_text = [mydict[char] for char in text]

# Create features and labels
nr_features = 4
X = []
y = []

for i in range(len(encoded_text) - nr_features):
    features = encoded_text[i : i + nr_features]
    X.append(features)
    label = encoded_text[i + nr_features]
    y.append(label)

# hot encode the labels
y = to_categorical(y, num_classes=len(mydict))

# build the model
model = Sequential(
    [
        Input(shape=(nr_features,)),
        Embedding(input_dim=len(mydict), output_dim=32, input_length=nr_features),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dense(len(mydict), activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
# Use callback to maximize accuracy
my_callback = keras.callbacks.EarlyStopping(
    patience=15,
    mode="max",
    monitor="accuracy",
    start_from_epoch=25,
    restore_best_weights=True,
)

# Train the model
model.fit(np.array(X), np.array(y), epochs=250, batch_size=2, callbacks=my_callback)

# Evaluate the model
eval_loss, eval_acc = model.evaluate(np.array(X), np.array(y))
print(f"Eval accuracy: {eval_acc:.4f}")
print(f"Eval loss: {eval_loss:.4f}")

# Save the model if it doesn't exist
filename = f"models/tiny_char_LM_train_acc-{eval_acc:.4f}.keras"
if not os.path.exists(filename):
    model.save(filename)

# predict the next character
sample_index = 21
data_for_eval = X[sample_index]
data_for_eval = np.array(data_for_eval).reshape(1, nr_features)

predicted = model.predict(data_for_eval, verbose=0)

# print the predicted character
pred_char = np.argmax(predicted, axis=1)
print(
    "Input:",
    "".join(idx2char[j] for j in X[sample_index]),
    "→ Predicted:",
    idx2char[pred_char[0]],
    " (Expected:",
    idx2char[encoded_text[sample_index + nr_features]] + ")",
)
