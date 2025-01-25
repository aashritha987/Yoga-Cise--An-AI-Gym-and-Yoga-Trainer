# YOGACISE: AI Powered Gym and Yoga Trainer

YOGACISE is an innovative application that leverages AI to provide real-time feedback on gym and yoga poses. By utilizing advanced technologies such as MediaPipe and OpenCV for video processing, this project aims to enhance the user experience by helping individuals improve their workout routines and achieve better results. This AI framework is able to Perform for 19 Gym Exercises and 19 Yoga Exercises.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)

## Features

- **Real-Time Pose Detection:** Uses MediaPipe for accurate detection of gym and yoga poses.
- **Personalized Feedback:** Provides tailored suggestions based on user performance.
- **User-Friendly Interface:** Developed using React for an intuitive user experience.
- **Cross-Platform Compatibility:** Accessible on various devices with a web browser.
- **Data Visualization:** Displays user progress over time with charts and statistics.

## Technologies Used

- **Frontend:** 
  - [React](https://reactjs.org/)
  - [Material-UI](https://mui.com/)
  
- **Backend:**
  - [Flask](https://flask.palletsprojects.com/)
  - [Flask-CORS](https://flask-cors.readthedocs.io/en/latest/) (for handling CORS issues)

- **Video Processing:**
  - [MediaPipe](https://google.github.io/mediapipe/)
  - [OpenCV](https://opencv.org/)


## Installation

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Node.js and npm
- Git

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GShankar555/YogaCise.git
   cd YOGACISE

2. **Set up the backend:**
    ```bash
    cd backend
    pip install -r requirements.txt

3. **Set up the frontend:**
    ```bash
    cd frontend
    npm install

4. **Run the backend server:**
    ```bash
    flask --app app run

5. **Run the frontend server:**
    ```bash
    npm start
 
**Home Page**

![Home page](https://github.com/GShankar555/YogaCise/blob/main/Home%20Page.png)


**Yoga Pose Correction**

![Yoga Interface](https://github.com/GShankar555/YogaCise/blob/main/Yoga.png)


**Gym Pose Correction**

![Yoga Interface](https://github.com/GShankar555/YogaCise/blob/main/Gym.png)
