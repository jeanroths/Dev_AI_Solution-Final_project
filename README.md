# SIMPATIA: Identification and Monitoring System for the Assured Protection of Workers with Artificial Intelligence

## Introduction

The **SIMPATIA** project aims to implement an advanced system for identifying and monitoring workers in Atvos plants, using artificial intelligence to ensure the safety and protection of employees. The system workflow is designed to integrate HikVision cameras, which capture real-time images of workers. These images are essential for training a detection model, which is fed with labeled data.

### System Workflow

1. **Image Capture**: HikVision cameras continuously capture images of workers in the plant. These images form the basis for training the artificial intelligence model.

2. **Model Training**: After capture, the images are labeled to identify specific characteristics, such as the use of Personal Protective Equipment (PPE). With these labeled images, the model is trained to recognize and monitor workers in real time.

3. **Real-Time Integration**: After training, the system is configured to connect the cameras to the monitoring code. This allows the model to make real-time predictions, identifying situations that may pose risks to worker safety.

4. **Cloud Storage**: Captured images and associated metadata are stored in Google Cloud Storage. This ensures that the data is secure and accessible for future analysis.

5. **Alerts and Reports**: When the system detects a critical situation, an alert or pop-up is generated in the HikCentral software. Additionally, relevant data is sent to a dashboard, where it can be viewed and analyzed by managers and safety teams.

### Project Structure

The project structure is organized as follows:

```
📦 Projeto_SIMPATIA
├── README.md
├── src
│ ├── annotations
│ ├── docs
│ ├── app
│ │ ├── bin_model
│ │ │ └── best.pt
│ │ ├── captures
│ │ │ └── capture_datetime.png
│ │ └── main.py
│ ├── features
│ ├── model_training
│ │ └── EPIs_yolo_detection_training.ipynb
│ └── requirements.txt
```

This project represents a significant advancement in the use of technology for workplace safety, promoting a safer and more efficient environment for all employees.

### Project Execution

#### 1. Installing Dependencies

Before running the code, it is necessary to install the required dependencies. Ensure that Python 3.7 or higher is installed on your system. You can install the dependencies using `pip`. Run the following command in the terminal in the `src` directory:

```bash
pip install -r requirements.txt
```

#### 2. Running the Code

After installing the dependencies, you can start the monitoring system by running the main script. In the terminal, navigate to the `src/app` directory and execute the following command:

```
python main.py
```

This will start the system, which will begin capturing webcam images and processing them in real time as previously described.

### Notes

- Ensure that the camera is connected and functioning correctly before running the code.
- Check if the necessary permissions to access the camera are granted.
- The system will generate screenshots and store them in the `captures` directory.
