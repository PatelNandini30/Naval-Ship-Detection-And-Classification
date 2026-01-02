# 🚢 Naval Ship Detection and Classification using YOLOv8

An AI-based **Naval Ship Detection and Classification System** built using **YOLOv8** and **Computer Vision** to automatically detect and classify naval vessels from **satellite, drone, and aerial imagery**.  
The system is designed for **real-time maritime surveillance, defense intelligence, and coastal security applications**.

---

## 📌 Project Overview

Maritime monitoring is critical for national security, naval defense, and border surveillance. Manual identification of ships from large-scale imagery is slow, error-prone, and inefficient.

This project leverages **deep learning-based object detection** using **YOLOv8** to:
- Detect naval ships in images and video streams
- Classify them into fine-grained vessel categories
- Operate efficiently in real-time environments

The model automatically learns spatial and structural features of ships such as shape, size, deck layout, and superstructure.

---

## ✨ Key Features

- 🚀 Real-time ship detection using YOLOv8
- 🛳️ Multi-class naval ship classification
- 📡 Works with satellite, drone, and aerial imagery
- 🧠 CNN-based automatic feature learning
- ⚡ High-speed inference with low latency
- 🌐 Web-based interface using Flask
- 📊 Performance evaluation using mAP, Precision, Recall

---

## 🧠 Supported Ship Classes

- Aircraft Carrier – Vikrant Class (IND)
- Aircraft Carrier – Kiev Class (IND)
- Aircraft Carrier – Fujian Class (CHI)
- Aircraft Carrier – Kuznetsov Class (CHI)
- Destroyer – Delhi Class (IND)
- Destroyer – Kolkata Class (IND)
- Destroyer – Rajput Class (IND)
- Corvette – Kora Class (IND)
- Corvette – Kamorta Class (IND)
- Corvette – Khukri Class (IND)
- Corvette – Veer Class (IND)
- Corvette – Azmat Class (PAK)
- Corvette – Babur Class (PAK)
- Corvette – Yarmook Class (PAK)
- Corvette – Jiangdao Class 056 (CHI)

> 🔹 The system can be easily extended to support additional ship categories.

---

## 🏗️ System Architecture

1. **Data Collection**
   - Satellite, drone, and aerial maritime images

2. **Data Preprocessing**
   - Image resizing, normalization, augmentation
   - Annotation in YOLO format

3. **Model Training**
   - YOLOv8 trained for multi-class object detection

4. **Inference Engine**
   - Detects ships and predicts class labels with confidence scores

5. **Deployment**
   - Flask-based web interface for image upload and detection

---

## 🛠️ Tech Stack

- **Python**
- **YOLOv8 (Ultralytics)**
- **PyTorch**
- **OpenCV**
- **NumPy**
- **Flask**
- **HTML / CSS**
- **CUDA (optional for GPU acceleration)**

---

## 📁 Project Structure

