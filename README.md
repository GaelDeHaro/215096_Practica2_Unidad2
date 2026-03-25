# 215096_Practica2_Unidad2
# Clasificador de Plantas con Vision Artificial

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=flat&logo=flask&logoColor=white)

Aplicación web local para identificar tipos de plantas a partir de imágenes usando **Transfer Learning** con MobileNetV2 y Keras. El usuario puede subir una foto y el modelo predice qué tipo de planta es junto con su porcentaje de confianza.

---

Este proyecto nació como una adaptación de un clasificador de imágenes original desarrollado en **PyTorch sobre Google Colab**, que identificaba niveles de sargazo en playas. El objetivo fue tomar ese conocimiento base de visión artificial y **reorientarlo hacia la clasificación de plantas**, haciendo además los siguientes cambios:

- **Migrar de PyTorch a TensorFlow/Keras**, para poder exportar el modelo en formato `.keras`
- **Salir de Google Colab** y construir una aplicación que corra completamente de forma **local** en cualquier computadora
- **Agregar una interfaz web amigable** donde cualquier persona pueda usar el modelo sin saber nada de programación
- Permitir **entrenar, exportar e importar** el modelo desde la misma interfaz


---

## ¿Qué hace el programa?

- **Identifica plantas** a partir de una imagen subida por el usuario
- **Muestra el porcentaje de confianza** de cada clase en un gráfico de barras
- **Entrena el modelo** con tus propias imágenes directamente desde la interfaz
- **Muestra el progreso del entrenamiento** en tiempo real (accuracy, loss, épocas)
- **Exporta el modelo** entrenado como archivo `.keras` para respaldo
- **Importa un modelo** previamente guardado sin necesidad de reentrenar
- **Guarda automáticamente** los mejores pesos durante el entrenamiento

---

## Plantas Soportadas

| Emoji | Planta | Carpeta |
|-------|--------|---------|
| 🌻 | Girasol | `girasol/` |
| 🌵 | Aloe Vera | `aloe_vera/` |
| 🌹 | Rosa | `rosa/` |
| 🌷 | Tulipán | `tulipan/` |
| 🌿 | Pasto | `pasto/` |

---

## 🧬 Arquitectura del Modelo

El modelo usa **Transfer Learning** con **MobileNetV2** como base preentrenada en ImageNet.

```
Entrada: imagen 224×224 px (RGB)
         │
         ▼
MobileNetV2 (preentrenada en ImageNet, 1.2M imágenes)
  Capas iniciales  →  detectan bordes y colores básicos
  Capas medias     →  detectan texturas y formas
  Capas finales    →  se ajustan a las plantas (fine-tuning)
         │
         ▼
GlobalAveragePooling2D
         │
         ▼
BatchNormalization
         │
         ▼
Dense(256, activación ReLU)
         │
         ▼
Dropout(0.4)  ←  evita sobreajuste
         │
         ▼
Dense(5, softmax)  ←  una probabilidad por planta
         │
         ▼
Salida: [girasol: 92%, aloe_vera: 3%, rosa: 2%, tulipan: 2%, pasto: 1%]
```

### Fases de Entrenamiento

El entrenamiento ocurre en dos fases:

**Fase 1 — Cabeza** (siempre se ejecuta)
Solo se entrenan las capas nuevas que se agregaron. La base de MobileNetV2 permanece congelada.

**Fase 2 — Fine-tuning** (opcional, recomendado)
Se descongelan las últimas 30 capas de MobileNetV2 y se ajustan con un learning rate 10 veces menor. Esto mejora el accuracy entre 5% y 10% adicional.

---

## ¿Cómo Funciona?

### Entrenamiento

```
Imágenes en carpetas
       │
       ▼
ImageDataGenerator  ←  aplica data augmentation (rotación, zoom, flip, etc.)
       │
       ▼
MobileNetV2 preentrenada
       │
       ▼
Función de pérdida: categorical_crossentropy
Optimizador: Adam
       │
       ▼
Mejor modelo guardado automáticamente → models/plant_model.keras
```

### Predicción

```
Imagen del usuario
       │
       ▼
Redimensionar a 224×224 px
       │
       ▼
Preprocesar (normalizar con MobileNetV2 preprocess_input)
       │
       ▼
Modelo cargado (.keras)
       │
       ▼
Vector de probabilidades [5 valores]
       │
       ▼
Clase con mayor probabilidad + porcentaje de confianza
```


## Estructura del Proyecto

```
/
│
├── app.py               ← servidor Flask + lógica del modelo
├── index.html           ← interfaz web completa
├── requirements.txt     ← dependencias de Python
├── README.md
│
├── models/              
│   └── plant_model.keras   ← modelo guardado tras entrenar
│
└── data/                
    ├── train/           ← 80% de las imágenes (para aprender)
    │   ├── girasol/
    │   ├── aloe_vera/
    │   ├── rosa/
    │   ├── tulipan/
    │   └── pasto/
    └── val/             
        ├── girasol/  ← 20% de las imágenes (para validar)
        ├── aloe_vera/
        ├── rosa/
        ├── tulipan/
        └── pasto/
```

---

## Instalación y Uso

### Requisitos

- Python 3.8 o superior

### 1. Clona el repositorio

```bash
git clone https://github.com/tu-usuario/215096_Practica2_Unidad2.git
cd plant-classifier
```

### 2. Crea un entorno virtual (recomendado)

```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Linux/Mac:
source venv/bin/activate
```

### 3. Instala las dependencias

```bash
pip install -r requirements.txt
```

### 4. Corre la aplicación

```bash
python app.py
```

### 5. Abre el navegador

```
http://127.0.0.1:5000
```

---

## 🖼️ Dataset Incluido

El proyecto ya cuenta con todas las imágenes necesarias para entrenar y validar el modelo. **En caso de querer agregar mas imagenes, se deberan poner en las carpetas segun su categoria.**

### Distribución del dataset

Cada categoría de planta cuenta con:

| Carpeta | Imágenes por clase | Uso |
|---------|-------------------|-----|
| `data/train/` | 100 fotos | Entrenamiento|
| `data/val/` | 20 fotos | Validación |

En total el dataset tiene **600 imágenes de entrenamiento** y **100 de validación**, distribuidas equitativamente entre las 5 clases.

> Las imágenes de `val` son completamente diferentes a las de `train`. Esto es fundamental para que la métrica de accuracy refleje qué tan bien generaliza el modelo con imágenes que nunca ha visto.

---

## Interfaz Web

La aplicación tiene 4 secciones accesibles desde pestañas:

### 🔍 Identificar
Arrastra o selecciona una foto de planta. El modelo muestra:
- El nombre de la planta detectada
- El porcentaje de confianza
- Un gráfico de barras con las probabilidades de todas las clases

### 🧠 Entrenar
Configura y ejecuta el entrenamiento:
- **Épocas**: número de vueltas sobre el dataset (recomendado: 20–30)
- **Batch Size**: imágenes procesadas por paso (recomendado: 16)
- **Learning Rate**: velocidad de aprendizaje (recomendado: 0.001)
- **Fine-tuning**: actívalo para mejor accuracy (tarda más)

Muestra progreso en tiempo real con barra, log y estadísticas por época.

### 💾 Modelo
- **Exportar**: descarga el `.keras` para guardar o compartir
- **Importar**: carga un `.keras` existente sin reentrenar

### 📋 Plantas
Guía de referencia con las clases y los detalles de la arquitectura.

---

## Tecnologías Utilizadas

| Tecnología | Uso |
|------------|-----|
| **TensorFlow / Keras** | Construcción y entrenamiento del modelo |
| **MobileNetV2** | Red base preentrenada (Transfer Learning) |
| **Flask** | Servidor web local y API REST |
| **Pillow** | Procesamiento de imágenes |
| **NumPy** | Operaciones numéricas |
| **HTML / CSS / JavaScript** | Interfaz web sin frameworks externos |

---

## Diferencias con el Código Original

Este proyecto es una evolución de un clasificador de sargazo hecho en PyTorch sobre Google Colab


---