# Blind Guide

The Blind Guide project aims to assist visually impaired individuals by utilizing computer vision to recognize objects and currencies in real-time. This repository includes two key models:

1. **Object Detection (YOLOv8)**: Identifies various objects from the COCO dataset to aid navigation.
2. **Currency Detection (SSD-MobileNet-v2-FPNLite-320)**: Recognizes different currency denominations, allowing independent financial transactions.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Acknowledgements](#acknowledgements)

## Features
- Real-time object detection and audio feedback.
- Currency recognition to facilitate monetary transactions.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/devkhedr/blind-guide.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the `object_detection_api.ipynb` for object detection and `currency_detector.ipynb` for currency detection.

## Usage
- Execute the Jupyter notebooks provided for each model to test and customize as needed.

## Models
- **YOLOv8**: Optimized for fast, accurate object detection.
- **SSD-MobileNet-v2-FPNLite-320**: Tailored for recognizing currency, with high accuracy in various conditions.

## Acknowledgements
This project was created as a senior project by students from Beni-Suef University, Egypt, under the guidance of Dr. Hossam Moftah.
