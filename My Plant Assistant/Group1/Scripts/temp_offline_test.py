import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the saved model
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 38)
model.load_state_dict(torch.load("disease_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Define transformation for the test image
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the list of classes
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# Function to predict the top 3 diseases for a plant
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    
    image_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top3_probs, top3_indices = torch.topk(probabilities, 3)
    top3_labels = [class_names[idx] for idx in top3_indices.tolist()]
    top3_probs = top3_probs.tolist()

    plant_diseases = defaultdict(list)
    for label, prob in zip(top3_labels, top3_probs):
        plant_name, disease = label.split("___")
        plant_diseases[plant_name].append((disease, prob))

    return plant_diseases, image

# Test the model with a new image
image_path = "10.jpg"
plant_diseases, image = predict_image(image_path)

fig = plt.figure(figsize=(8, 10))
gs = GridSpec(2, 1, height_ratios=[3, 1])

ax_image = fig.add_subplot(gs[0])
ax_image.imshow(image)
ax_image.axis("off")

ax_text = fig.add_subplot(gs[1])
ax_text.axis("off")

# Display predictions
text = ""
for plant, diseases in plant_diseases.items():
    text += f"Plant: {plant}\n"
    for disease, prob in diseases:
        text += f" - {disease}: {prob * 100:.2f}%\n"

ax_text.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, wrap=True)

plt.tight_layout()
plt.show()

