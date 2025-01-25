import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFont, ImageDraw
import requests
import cv2
import numpy as np
import pickle
import arabic_reshaper
from bidi.algorithm import get_display

model_load_path = "flower_model.pth"
labels_load_path = "flower_labels.pkl"

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load class labels
with open(labels_load_path, "rb") as f:
    idx_to_class = pickle.load(f)
print(f"Loaded {len(idx_to_class)} class labels.")

# Data transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(idx_to_class))
model.load_state_dict(torch.load(model_load_path))
model = model.to(device)
model.eval()

# Function to fetch a single frame from IP Webcam
def fetch_frame(url):
    try:
        response = requests.get(url, stream=True)
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error fetching frame: {e}")
        return None

# Function to predict the flower class of a frame
def predict_frame(frame):
    try:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = probabilities.max(1)

            top_prob = top_prob.item()
            top_class = top_class.item()

            if top_prob > 0.7:
                class_name = idx_to_class[top_class]
                return f" این یک \"{class_name}\" است"
            else:
                return "گلی یافت نشد"
    except Exception as e:
        print(f"Error processing frame: {e}")
        return "گلی یافت نشد"

# Function to add Persian text
def add_persian_text(frame, text, position, font_path="arial.ttf", font_size=32, color=(0, 255, 0)):
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        font = ImageFont.truetype(font_path, font_size)

        draw = ImageDraw.Draw(frame_pil)
        draw.text(position, bidi_text, font=font, fill=color)

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        print(f"Error adding Persian text: {e}")
        return frame

# Live prediction using phone's camera
def live_prediction(url, font_path="arial.ttf"):
    print("Starting live prediction. Press 'q' to quit.")
    while True:
        frame = fetch_frame(url)
        if frame is None:
            print("Error fetching frame. Retrying...")
            continue

        message = predict_frame(frame)

        frame = add_persian_text(frame, message, position=(10, 50), font_path=font_path, font_size=32)

        cv2.imshow("Flower Identifier - Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    ip_webcam_url = "http://192.168.164.155:8080/shot.jpg"
    font_path = "IRANSans.ttf"
    live_prediction(ip_webcam_url, font_path=font_path)
