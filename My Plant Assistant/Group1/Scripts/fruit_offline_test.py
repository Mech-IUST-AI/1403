import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import ImageFont, ImageDraw, Image
import pickle
import cv2
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display

# Model and labels path
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

# Function to classify objects in a saved image
def classify_image(image_path, probability_threshold=0.7):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classification_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = probabilities.max(1)

        top_prob = top_prob.item()
        top_class = top_class.item()

        if top_prob > probability_threshold:
            class_name = class_labels[top_class]
            return f"این یک {class_name} است"
        else:
            return "میوه ای یافت نشد"

# Function to add Persian text with limited width
def add_persian_text(frame, text, font_path="IRANSans.ttf", font_size=30):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)

    image_width = pil_image.width

    max_text_width = image_width // 2
    text_bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]

    while text_width > max_text_width:
        font_size -= 2
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.textbbox((0, 0), bidi_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]

    position = ((image_width - text_width) // 2, 20)
    draw.text(position, bidi_text, font=font, fill=(0, 255, 0))

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Main function to test the model on a saved image
def test_saved_image(image_path, font_path = "IRANSans.ttf"):
    print(f"Testing image: {image_path}")
    
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not load the image. Check the file path.")
        return

    message = classify_image(image_path)

    frame = add_persian_text(frame, message, font_path=font_path, font_size=32)

    window_name = "Fruit Identifier"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, frame.shape[1], frame.shape[0])

    cv2.imshow(window_name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "13.jfif"
    persian_font_path = "IRANSans.ttf"
    test_saved_image(image_path, persian_font_path)