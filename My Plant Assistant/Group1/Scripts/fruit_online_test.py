import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import ImageFont, ImageDraw, Image
import pickle
import requests
import cv2
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display

model_load_path = "fruit_model.pth"
labels_load_path = "fruit_labels.pkl"

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load class labels
with open(labels_load_path, 'rb') as f:
    class_labels = pickle.load(f)
num_classes = len(class_labels)
print(f"Loaded {num_classes} class labels.")

# Data transforms for classification
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the classification model
classification_model = models.resnet18(pretrained=False)
classification_model.fc = nn.Linear(classification_model.fc.in_features, num_classes)
classification_model.load_state_dict(torch.load(model_load_path))
classification_model = classification_model.to(device)
classification_model.eval()

# Load the object detection model (Faster R-CNN)
detection_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
detection_model = detection_model.to(device)
detection_model.eval()

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

# Function to predict objects in a frame
def predict_objects(frame):

    image_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        detections = detection_model(image_tensor)[0]

    return detections

# Function to classify objects
def classify_objects(frame, detections, threshold=0.5, probability_threshold = 0.7):
    results = []
    for i, box in enumerate(detections['boxes']):

        if detections['scores'][i] >= threshold:

            x1, y1, x2, y2 = map(int, box.tolist())

            crop = frame[y1:y2, x1:x2]

            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = classification_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_class = probabilities.max(1)

                top_prob = top_prob.item()
                top_class = top_class.item()

                if top_prob > probability_threshold:
                    class_name = class_labels[top_class]
                    results.append((class_name, (x1, y1, x2, y2)))

    return results

FRUIT = {
    "موز": "موز انواع",
    "سیب": "سیب دورنگ تابستانه",
    "هندوانه": "هندوانه انواع",
    "اسفناج": "اسفناج",
    "انار": "انار",
    "آناناس": "آناناس طلایی",
    "انگور": "انگور انواع",
    "بادمجان": "بادمجان",
    "پرتغال": "پرتقال انواع",
    "پیاز": "پیاز زرد",
    "تربچه": "تربچه",
    "خیار": "خیار سالادی",
    "ذرت": "ذرت شیرین",
    "زنجبیل": "زنجبیل تازه",
    "سویا": "سویا",
    "سیب زمینی": "سیب زمینی تازه",
    "سیر": "سیر خشک",
    "شلغم": "شلغم",
    "فلفل": "فلفل ریز تند",
    "فلفل دلمه": "فلفل دلمه رنگی",
    "کاهو": "کاهو پیچ (سالادی)",
    "کلم": "کلم سفید",
    "کیوی": "کیوی",
    "گل کلم": "گل کلم",
    "گلابی": "گلابی انواع",
    "گوجه": "گوجه فرنگی بوته",
    "چغندر": "چغندر برش",
    "مانگو": "مانگو",
    "نخود فرنگی": "نخود فرنگی",
    "هندوانه": "هندوانه انواع",
    "هویج": "هویج فرنگی و ایرانی"
}

# Function to fetch fruit price from the API
def fetch_fruit_price(fruit_name):
    url = "https://sarvban.com/market/api/fruitdata"
    data = {
        "prod_id": fruit_name,
        "prod_type": fruit_name
    }

    try:
        response = requests.post(url=url, json=data)
        response.raise_for_status()
        price_data = response.json().get("minPrice_data", [])
        if price_data:
            return price_data[-1]
        else:
            return "قیمت موجود نیست"
    except Exception as e:
        print(f"Error fetching price for {fruit_name}: {e}")
        return "خطا در بازیابی قیمت"

# Function to add Persian text
def add_persian_text(frame, text, position, font_path="path_to_persian_font.ttf", font_size=30):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, bidi_text, font=font, fill=(0, 255, 0))

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Function to draw rounded rectangles
def draw_rounded_rectangle(image, top_left, bottom_right, color, thickness=2, radius=10):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.circle(image, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(image, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(image, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(image, (x2 - radius, y2 - radius), radius, color, thickness)

    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

# live prediction function
def live_prediction_with_price(url, font_path):
    print("Starting live prediction with price reporting. Press 'q' to quit.")
    while True:
        frame = fetch_frame(url)
        if frame is None:
            print("Error fetching frame. Retrying...")
            continue

        detections = predict_objects(frame)
        predictions = classify_objects(frame, detections)

        if not predictions:
            frame = add_persian_text(frame, "میوه ای یافت نشد", (50, 50), font_path=font_path, font_size=32)
        else:
            for class_name, (x1, y1, x2, y2) in predictions:
                name = FRUIT.get(class_name)
                fruit_price = fetch_fruit_price(name)
                persian_text = f"این یک {class_name} است - قیمت: {fruit_price}"

                draw_rounded_rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2, radius=10)
                frame = add_persian_text(frame, persian_text, (x1, y1 - 30), font_path=font_path, font_size=24)

        frame = add_persian_text(frame, "Fruit Identifier", (10, 10), font_path=font_path, font_size=32)
        cv2.imshow("Fruit Identifier with Price", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    ip_webcam_url = "http://192.168.164.155:8080/shot.jpg"
    persian_font_path = "IRANSans.ttf"
    live_prediction_with_price(ip_webcam_url, persian_font_path)