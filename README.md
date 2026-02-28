# Age and Gender Detection System

A Computer Vision-based application that predicts a person’s age group and gender from facial images using pre-trained deep learning models integrated with OpenCV.

This project demonstrates practical implementation of deep learning model inference, image preprocessing, and real-time face analysis using Python.

---

## Project Overview

The system detects human faces in an image and estimates:

- Gender (Male/Female)  
- Age Range (e.g., 0–2, 4–6, 8–12, 15–20, etc.)

It uses pre-trained Caffe deep learning models for classification and integrates them into a Python-based inference pipeline.

---

## Tech Stack

- Python  
- OpenCV  
- Deep Learning (Caffe Models)  

**Pre-trained Models:**
- `age_net.caffemodel`
- `gender_net.caffemodel`

**Model Architecture Files:**
- `deploy_age.prototxt`
- `deploy_gender.prototxt`

---

## How It Works

1. Detects faces in the input image using OpenCV.  
2. Extracts the face region (ROI).  
3. Preprocesses the image into a blob for model input.  
4. Performs inference using:
   - Age classification model  
   - Gender classification model  
5. Displays predicted age range and gender on the image.

This project highlights understanding of:

- Image preprocessing  
- Neural network inference  
- Model integration  
- Computer vision pipeline design  

---

## Project Structure

```
age-and-gender-detector/
│
├── age_detector.py
├── age_net.caffemodel
├── gender_net.caffemodel
├── deploy_age.prototxt
├── deploy_gender.prototxt
```

---

## Installation & Usage

### Clone Repository
```
git clone https://github.com/goranivishakha/age-and-gender-detector.git
cd age-and-gender-detector
```

### Install Dependencies
```
pip install opencv-python numpy
```

### Run the Project
```
python age_detector.py
```

---

## Key Learnings

- Practical implementation of pre-trained deep learning models  
- Working with model configuration (`.prototxt`) and weights (`.caffemodel`)  
- Real-time image processing using OpenCV  
- Integration of AI models into production-style Python scripts  

---

## Future Improvements

- Convert into a Flask/FastAPI web application  
- Deploy as a cloud-based ML inference service  
- Improve accuracy using modern CNN architectures  

---

## Author

**Vishakha Gorani**  
Cloud & Cybersecurity Enthusiast | Strong in DSA & Problem Solving
