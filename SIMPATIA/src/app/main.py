import cv2
import os
from ultralytics import YOLO
import time

# Load the model
def load_model(model_path):
    return YOLO(model_path)

# Setting up the colors
def setup_colors():
    return {
        "head": (0, 0, 255),    # red for "head" (no helmet)
        "person": (255, 0, 0),  # blue for "person"
    }

# Creates the directory for capture
def create_capture_directory(directory):
    os.makedirs(directory, exist_ok=True)

# Capture and processing video frames
def capture_video(model, colors, save_dir, capture_interval):
    url_rtsp = 0 # "rtsp://USUARIO:SENHA@DOMINIO:PORTA/h264/ch1/main/av_stream" - this is the RTSP URL pattern address to access HikVision video frames in realtime
    cap = cv2.VideoCapture(url_rtsp)
    
    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        return

    last_capture_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Fim do vídeo. Reiniciando...")
            cap = cv2.VideoCapture(url_rtsp)
            continue
        
        results = model.predict(frame)
        show_frame = True  # Defines if frame is showed or not

        for bbox in results[0].boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            conf = bbox.conf[0]
            cls = int(bbox.cls[0])
            class_name = model.names[cls]
            
            if class_name in colors:  # Only the relevant classes
                color = colors[class_name]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Verify if the classes "head" and "person" appears without helmet
            if class_name == "head":
                show_frame = True  # Only activate if "head" is detected without helmet
            
            if class_name == "person" and time.time() - last_capture_time > capture_interval and show_frame:
                filename = os.path.join(save_dir, f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(filename, frame)
                print(f"Imagem salva em: {filename}")
                last_capture_time = time.time()
        
        if show_frame:  # Only show the frame if the condition is atendded
            cv2.imshow('Detecção', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    model_path = "bin_model/best.pt"
    save_dir = "captures"
    capture_interval = 5  # Captures each 5 seconds

    model = load_model(model_path)
    colors = setup_colors()
    create_capture_directory(save_dir)
    capture_video(model, colors, save_dir, capture_interval)

if __name__ == "__main__":
    main()