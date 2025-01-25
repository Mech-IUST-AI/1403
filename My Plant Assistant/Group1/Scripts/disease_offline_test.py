import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict

# Define device
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

# Define the solutions
disease_solutions = {
    "Apple___Apple_scab": "Solution: Use fungicides like captan or mancozeb, prune infected leaves, and avoid overhead watering.",
    "Apple___Black_rot": "Solution: Remove and destroy infected fruits, prune infected branches, and apply copper-based fungicides.",
    "Apple___Cedar_apple_rust": "Solution: Remove nearby cedar trees, prune infected parts, and use fungicides containing myclobutanil.",
    "Apple___healthy": "Solution: Your apple tree is healthy! Maintain proper watering and fertilization practices.",
    "Blueberry___healthy": "Solution: Your blueberry plant is healthy! Keep up with proper watering and fertilization.",
    "Cherry_(including_sour)___healthy": "Solution: Your cherry tree is healthy! Ensure proper pruning and watering.",
    "Cherry_(including_sour)___Powdery_mildew": "Solution: Prune infected branches, improve air circulation, and apply fungicides like sulfur or potassium bicarbonate.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Solution: Use resistant hybrids, rotate crops, and apply fungicides like azoxystrobin.",
    "Corn_(maize)___Common_rust_": "Solution: Plant resistant varieties and apply fungicides like propiconazole if needed.",
    "Corn_(maize)___healthy": "Solution: Your corn plant is healthy! Ensure proper crop rotation and fertilization.",
    "Corn_(maize)___Northern_Leaf_Blight": "Solution: Use resistant hybrids, practice crop rotation, and apply fungicides like pyraclostrobin if severe.",
    "Grape___Black_rot": "Solution: Apply fungicides like myclobutanil or mancozeb, remove mummified berries, and prune infected leaves.",
    "Grape___Esca_(Black_Measles)": "Solution: Prune infected parts and avoid wounding the vines. Apply fungicides if necessary.",
    "Grape___healthy": "Solution: Your grape plant is healthy! Maintain proper watering and fertilization practices.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Solution: Prune infected leaves and apply fungicides like mancozeb or copper oxychloride.",
    "Orange___Haunglongbing_(Citrus_greening)": "Solution: Remove infected trees, control psyllid vectors with insecticides, and plant resistant varieties.",
    "Peach___Bacterial_spot": "Solution: Apply copper-based bactericides, prune infected twigs, and plant resistant varieties.",
    "Peach___healthy": "Solution: Your peach tree is healthy! Ensure regular watering and fertilization.",
    "Pepper,_bell___Bacterial_spot": "Solution: Use copper-based sprays, remove infected leaves, and rotate crops regularly.",
    "Pepper,_bell___healthy": "Solution: Your bell pepper plant is healthy! Keep up with proper watering and pest management.",
    "Potato___Early_blight": "Solution: Use fungicides like chlorothalonil, remove infected plants, and rotate crops.",
    "Potato___healthy": "Solution: Your potato plant is healthy! Maintain proper watering and pest control practices.",
    "Potato___Late_blight": "Solution: Remove infected plants and apply fungicides like mancozeb or chlorothalonil.",
    "Raspberry___healthy": "Solution: Your raspberry plant is healthy! Ensure proper pruning and fertilization.",
    "Soybean___healthy": "Solution: Your soybean plant is healthy! Keep up with proper watering and pest management.",
    "Squash___Powdery_mildew": "Solution: Apply fungicides like sulfur or potassium bicarbonate, and improve air circulation by pruning overcrowded leaves.",
    "Strawberry___healthy": "Solution: Your strawberry plant is healthy! Ensure proper watering and pest control.",
    "Strawberry___Leaf_scorch": "Solution: Remove infected leaves, improve air circulation, and use fungicides like myclobutanil.",
    "Tomato___Bacterial_spot": "Solution: Use copper-based sprays, remove infected leaves, and practice crop rotation.",
    "Tomato___Early_blight": "Solution: Remove infected leaves, rotate crops, and use fungicides like chlorothalonil or copper sprays.",
    "Tomato___healthy": "Solution: Your tomato plant is healthy! Keep up with proper watering and pest management.",
    "Tomato___Tomato_mosaic_virus": "Solution: Remove infected plants, avoid handling healthy plants after touching infected ones, and sterilize tools regularly.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Solution: Control whiteflies with insecticidal sprays and plant virus-resistant varieties."
}

# Combined function to predict and display
def predict_and_display(image_path):

    image = Image.open(image_path).convert("RGB")
    image_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_prob, top_idx = torch.max(probabilities, 0)
    top_label = class_names[top_idx.item()]
    plant_name, disease = top_label.split("___")

    solution = disease_solutions.get(top_label, "Solution: Not available for this disease.")

    fig = plt.figure(figsize=(8, 12))
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    ax_image = fig.add_subplot(gs[0])
    ax_image.imshow(image)
    ax_image.axis("off")

    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")

    text = f"Plant: {plant_name}\nDisease: {disease}\n{solution}"
    ax_text.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, wrap=True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "10.jpg"
    predict_and_display(image_path)
