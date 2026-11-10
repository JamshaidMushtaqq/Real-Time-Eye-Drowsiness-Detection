# Real-Time-Eye-Drowsiness-Detection
# 🚗 AI-Based Driver Drowsiness Detection System

## 📌 Project Overview

The **AI-Based Driver Drowsiness Detection System** is a real-time safety application developed using Python, Deep Learning, and Computer Vision.

The system monitors a driver's eyes through a webcam and detects whether the eyes are open or closed. If the driver's eyes remain closed for 3 seconds, the system automatically triggers an emergency alarm to alert the driver and help prevent potential road accidents.

## ✨ Features

- Real-time eye monitoring using a webcam.
- AI-based eye state detection (Open / Closed).
- CNN (Convolutional Neural Network) deep learning model.
- Face and eye landmark detection using MediaPipe.
- Automatic emergency alarm system.
- Real-time drowsiness detection.
- Visual status display.

## 🛠️ Technologies Used

- **Python** – Main programming language.
- **TensorFlow / Keras** – Deep learning model.
- **OpenCV** – Image processing and webcam access.
- **MediaPipe** – Face and eye landmark detection.
- **NumPy** – Numerical operations.
- **Scikit-learn** – Model evaluation and dataset splitting.
- **Matplotlib & Seaborn** – Training visualization.

## ⚙️ How It Works

1. The webcam captures the driver's face in real time.
2. MediaPipe detects facial landmarks and identifies eye regions.
3. The CNN model analyzes eye images.
4. The system classifies eyes as Open or Closed.
5. If the eyes remain closed for 3 seconds, an emergency alarm is triggered.

## 📂 Project Structure

```text
AI-Driver-Drowsiness-Detection/
│
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   └── webcam.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── best_eye_cnn.keras
│   └── class_names.json
│
├── outputs/
│
├── emergency_alarm.wav
├── requirements.txt
└── README.md
```

## 🚀 Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/AI-Driver-Drowsiness-Detection.git
```

**2. Open the project folder**

```bash
cd AI-Driver-Drowsiness-Detection
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Prepare the dataset**

Place the MRL Eye Dataset images inside `data/raw/`, then run:

```bash
python src/prepare_data.py
```

**5. Train the CNN model**

```bash
python src/train.py
```

This generates the trained model and class names inside the `models/` directory.

**6. Run the project**

```bash
python src/webcam.py
```

Make sure your webcam is connected and the trained model is available.

## 🎯 Project Objective

The main objective of this project is to improve driver safety by using Artificial Intelligence to detect drowsiness in real time and provide an early warning through an emergency alarm.

## 👨‍💻 Developer

Developed as an AI / Machine Learning project.

## ⭐ Support

If you find this project interesting, consider giving this repository a star on GitHub!

---

**Note:** This system is a prototype for driver-safety assistance and should not replace proper driver alertness or road-safety practices. The current alarm implementation uses Windows `winsound`.
