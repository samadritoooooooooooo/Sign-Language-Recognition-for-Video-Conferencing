import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

SIGNS     = ['yes', 'no', 'question', 'repeat', 'ok', 'wait']
SEQUENCES = 120
FRAMES    = 30
DATA_DIR  = 'ISL_Data'


print("Loading data...")

X = []
y = []

for label_idx, sign in enumerate(SIGNS):
    clips_loaded = 0
    for seq in range(SEQUENCES):
        frames = []
        valid = True
        for frame_num in range(FRAMES):
            path = os.path.join(DATA_DIR, sign, str(seq), f'{frame_num}.npy')
            if not os.path.exists(path):
                valid = False
                break
            frames.append(np.load(path))

        if valid:
            X.append(frames)
            y.append(label_idx)
            clips_loaded += 1

    print(f'  {sign}: {clips_loaded} clips loaded')

X = np.array(X)
y = np.array(y)

print(f'\nDataset shape: {X.shape}')
print(f'Labels shape:  {y.shape}')

y_cat = to_categorical(y, num_classes=len(SIGNS))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

print(f'\nTraining samples: {X_train.shape[0]}')
print(f'Testing samples:  {X_test.shape[0]}')


model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(FRAMES, 63)),
    Dropout(0.2),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(SIGNS), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

print('\nTraining...')
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr]
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f' Test Accuracy: {accuracy * 100:.2f}%')

model.save('isl_model.keras')

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()