# Familial Adenomatous Polyposis Detection

## Overview
This Python web application is designed to assist medical professionals by automatically detecting and counting polyps from colonoscopy videos. It employs machine learning models to analyze video frames and identify potential polyps, providing a count of these features which can be crucial for diagnosing Familial Adenomatous Polyposis.

## Features
- **Video Upload**: Users can upload colonoscopy video files directly through the web interface.
- **Polyp Detection**: Utilizes pre-trained machine learning models to detect polyps in video frames.
- **Result Display**: Shows annotated video frames with detected polyps highlighted and counts the total number of detected polyps.
- **Interactive UI**: A user-friendly interface that allows easy navigation and operation by end-users.

## Technology Stack
- **Flask**: Serves the backend and handles all server-side logic.
- **Python**: For running machine learning models and processing data.
- **HTML/CSS/JavaScript**: Frontend presentation and interaction.
- **Bootstrap**: For responsive design.

## Installation

To get this project up and running on your local machine, follow these steps:

1. **Run steup_ev.sh**
2. **Set up local path in model files, located in modesl directory**
3. **Start the Flask server**

This will start the local server on `http://127.0.0.1:5000/`.

## Usage

1. **Open a web browser** and navigate to `http://127.0.0.1:5000/`.
2. **Upload a video** by selecting 'Choose File' and then click 'Upload Video'.
3. **Select a detection model** from the dropdown menu and click 'Count Polyps'.
4. **View results** on the same page, including the annotated images and the polyp count.


## Contact
Your Name – singh.shivam311095@gmail.com



