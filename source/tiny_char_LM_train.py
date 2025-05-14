from icecream import ic
from tensorflow.keras.models import Sequential
import keras
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
myrevdict = dict(zip(range(len(mysorted)), mysorted))

ic(myrevdict)

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
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Use callback to maximize accuracy
my_callback = keras.callbacks.EarlyStopping(
    patience=15,
    mode="max",
    monitor="accuracy",
    start_from_epoch=25,
    restore_best_weights=True,
)

# Train the model
model.fit(np.array(X), np.array(y), epochs=250, batch_size=4, callbacks=my_callback)

# Evaluate the model
eval_loss, eval_acc = model.evaluate(np.array(X), np.array(y))
print(f"Eval accuracy: {eval_acc:.4f}")
print(f"Eval loss: {eval_loss:.4f}")

# predict the next character
sequence_index = 10
data_for_eval = X[sequence_index]
data_for_eval = np.array(data_for_eval).reshape(1, nr_features)

predicted = model.predict(data_for_eval)
predicted = np.argmax(predicted, axis=1)

print(f"Predicted: {myrevdict[predicted[0]]}")
print(f"Expected: {myrevdict[encoded_text[sequence_index + nr_features]]}")
