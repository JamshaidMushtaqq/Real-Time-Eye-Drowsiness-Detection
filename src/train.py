from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 20
SEED = 42

def make_dataset(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED
    )

train_ds = make_dataset("train", True)
val_ds = make_dataset("val", False)
test_ds = make_dataset("test", False)
class_names = train_ds.class_names
print("Class order:", class_names)
with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
normalizer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (normalizer(x), y)).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalizer(x), y)).prefetch(AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (normalizer(x), y)).prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.04),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomContrast(0.10),
], name="augmentation")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 1)),
    data_augmentation,
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.15),

    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "best_eye_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=150)
plt.close()

# Evaluate best checkpoint on held-out test set
best_model = tf.keras.models.load_model(MODEL_DIR / "best_eye_cnn.keras")
test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)
y_true = np.concatenate([y.numpy() for _, y in test_ds])
probs = best_model.predict(test_ds, verbose=1)
y_pred = np.argmax(probs, axis=1)

report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\nTEST REPORT\n", report)
(OUT_DIR / "classification_report.txt").write_text(
    f"Test loss: {test_loss:.4f}\nTest accuracy: {test_acc:.4f}\n\n{report}",
    encoding="utf-8"
)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

print(f"\nBest model: {MODEL_DIR / 'best_eye_cnn.keras'}")
print(f"Test accuracy: {test_acc:.4f}")
