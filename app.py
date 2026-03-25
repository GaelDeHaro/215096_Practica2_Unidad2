"""
PlantAI - Clasificador de Plantas Local
Flask backend con TensorFlow/Keras + MobileNetV2
"""

import os
import io
import json
import time
import threading
import base64
from pathlib import Path

from flask import Flask, request, jsonify, send_file, Response, render_template_string
import numpy as np
from PIL import Image

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ─── Configuración global ────────────────────────────────────────────────────
CLASSES      = ['aloe_vera', 'girasol', 'pasto', 'rosa', 'tulipan']
NUM_CLASSES  = len(CLASSES)
IMG_SIZE     = (224, 224)
DATA_DIR     = Path('data')
MODELS_DIR   = Path('models')
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH   = MODELS_DIR / 'plant_model.keras'

app = Flask(__name__)

# Estado global del entrenamiento
training_state = {
    'running':   False,
    'progress':  0,
    'epoch':     0,
    'total_epochs': 0,
    'train_acc': 0.0,
    'val_acc':   0.0,
    'train_loss': 0.0,
    'val_loss':   0.0,
    'log':       [],
    'done':      False,
    'error':     None,
}

# Modelo cargado en memoria
current_model = None

# ─── Utilidades ──────────────────────────────────────────────────────────────

def load_model_from_disk():
    """Carga el modelo .keras si existe."""
    global current_model
    if MODEL_PATH.exists():
        current_model = tf.keras.models.load_model(str(MODEL_PATH))
        print(f"[PlantAI] Modelo cargado desde {MODEL_PATH}")
        return True
    return False


def build_model():
    """Construye MobileNetV2 con Transfer Learning para las 5 plantas."""
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False   # Congelar base primero

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def preprocess_image(image_bytes):
    """Preprocesa bytes de imagen para inferencia."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# ─── Callback de progreso ────────────────────────────────────────────────────

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        training_state['epoch']      = epoch + 1
        training_state['progress']   = int(((epoch + 1) / self.total_epochs) * 100)
        training_state['train_acc']  = round(float(logs.get('accuracy', 0)) * 100, 2)
        training_state['val_acc']    = round(float(logs.get('val_accuracy', 0)) * 100, 2)
        training_state['train_loss'] = round(float(logs.get('loss', 0)), 4)
        training_state['val_loss']   = round(float(logs.get('val_loss', 0)), 4)
        training_state['log'].append(
            f"Época {epoch+1}/{self.total_epochs} — "
            f"Train Acc: {training_state['train_acc']}% | "
            f"Val Acc: {training_state['val_acc']}% | "
            f"Loss: {training_state['train_loss']}"
        )


# ─── Hilo de entrenamiento ───────────────────────────────────────────────────

def run_training(epochs, batch_size, learning_rate, fine_tune):
    global current_model, training_state

    try:
        training_state.update({
            'running': True, 'done': False, 'error': None,
            'progress': 0, 'epoch': 0, 'total_epochs': epochs,
            'log': ['Preparando datos...']
        })

        # Data augmentation para entrenamiento
        train_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_data = train_gen.flow_from_directory(
            DATA_DIR / 'train',
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode='categorical',
            classes=CLASSES,
            shuffle=True
        )
        val_data = val_gen.flow_from_directory(
            DATA_DIR / 'val',
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode='categorical',
            classes=CLASSES,
            shuffle=False
        )

        training_state['log'].append(
            f"Datos cargados: {train_data.samples} entrenamiento, {val_data.samples} validación"
        )

        # Construir o usar modelo existente
        model = build_model()
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            ProgressCallback(epochs),
            ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor='val_accuracy', verbose=0),
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=0)
        ]

        training_state['log'].append('Iniciando entrenamiento (fase 1: cabeza)...')
        model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0
        )

        # Fine-tuning opcional: descongelar últimas 30 capas de la base
        if fine_tune:
            training_state['log'].append('Fine-tuning: descongelando capas base...')
            base_model = model.layers[0]
            base_model.trainable = True
            for layer in base_model.layers[:-30]:
                layer.trainable = False

            model.compile(
                optimizer=optimizers.Adam(learning_rate=learning_rate / 10),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            ft_epochs = max(5, epochs // 2)
            training_state['total_epochs'] = epochs + ft_epochs
            model.fit(
                train_data,
                validation_data=val_data,
                epochs=ft_epochs,
                callbacks=callbacks,
                verbose=0
            )

        model.save(str(MODEL_PATH))
        current_model = model

        training_state['done']     = True
        training_state['running']  = False
        training_state['progress'] = 100
        training_state['log'].append(f'✅ Entrenamiento completado. Modelo guardado en {MODEL_PATH}')

    except Exception as e:
        training_state['error']   = str(e)
        training_state['running'] = False
        training_state['done']    = True
        training_state['log'].append(f'❌ Error: {e}')


# ─── Rutas de la API ─────────────────────────────────────────────────────────

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/api/predict', methods=['POST'])
def predict():
    if current_model is None:
        return jsonify({'error': 'No hay modelo cargado. Entrena o importa uno primero.'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No se recibió imagen.'}), 400

    file = request.files['image']
    img_bytes = file.read()

    try:
        arr = preprocess_image(img_bytes)
        preds = current_model.predict(arr, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx]) * 100

        results = [
            {'class': CLASSES[i], 'confidence': round(float(preds[i]) * 100, 2)}
            for i in range(NUM_CLASSES)
        ]
        results.sort(key=lambda x: x['confidence'], reverse=True)

        # Thumbnail en base64 para devolver al frontend
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img.thumbnail((300, 300))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        thumb_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'prediction':  CLASSES[class_idx],
            'confidence':  round(confidence, 2),
            'all_classes': results,
            'thumbnail':   thumb_b64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def start_training():
    if training_state['running']:
        return jsonify({'error': 'Ya hay un entrenamiento en curso.'}), 409

    body = request.get_json() or {}
    epochs        = int(body.get('epochs', 20))
    batch_size    = int(body.get('batch_size', 16))
    learning_rate = float(body.get('learning_rate', 0.001))
    fine_tune     = bool(body.get('fine_tune', True))

    if not (DATA_DIR / 'train').exists():
        return jsonify({'error': f'No se encontró la carpeta {DATA_DIR}/train'}), 400

    t = threading.Thread(
        target=run_training,
        args=(epochs, batch_size, learning_rate, fine_tune),
        daemon=True
    )
    t.start()
    return jsonify({'message': 'Entrenamiento iniciado.'})


@app.route('/api/training_status')
def training_status():
    return jsonify(training_state)


@app.route('/api/export')
def export_model():
    if not MODEL_PATH.exists():
        return jsonify({'error': 'No hay modelo guardado.'}), 404
    return send_file(
        str(MODEL_PATH),
        as_attachment=True,
        download_name='plant_model.keras',
        mimetype='application/octet-stream'
    )


@app.route('/api/import', methods=['POST'])
def import_model():
    global current_model
    if 'model' not in request.files:
        return jsonify({'error': 'No se recibió archivo.'}), 400

    file = request.files['model']
    if not file.filename.endswith('.keras'):
        return jsonify({'error': 'El archivo debe tener extensión .keras'}), 400

    tmp_path = MODELS_DIR / 'uploaded_model.keras'
    file.save(str(tmp_path))

    try:
        loaded = tf.keras.models.load_model(str(tmp_path))
        tmp_path.rename(MODEL_PATH)
        current_model = loaded
        return jsonify({'message': 'Modelo importado correctamente.'})
    except Exception as e:
        return jsonify({'error': f'No se pudo cargar el modelo: {e}'}), 500


@app.route('/api/model_info')
def model_info():
    if current_model is None:
        return jsonify({'loaded': False})
    return jsonify({
        'loaded':     True,
        'classes':    CLASSES,
        'input_size': list(IMG_SIZE),
        'path':       str(MODEL_PATH) if MODEL_PATH.exists() else None
    })


# ─── Inicio ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n🌿 PlantAI — Clasificador de Plantas")
    print("=" * 40)
    if load_model_from_disk():
        print("✅ Modelo cargado automáticamente.")
    else:
        print("⚠️  No hay modelo guardado. Entrena uno primero.")
    print(f"Clases: {', '.join(CLASSES)}")
    print(f"Abre tu navegador en: http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000, threaded=True)
