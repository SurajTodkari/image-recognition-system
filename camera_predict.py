import cv2
import torch
from torchvision import transforms
from model import SimpleCNN  # ✅ Updated import
from PIL import Image

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Define the transform to match training
transform = transforms.Compose([
    transforms.Resize((28, 28)),         # Resize image to match model input
    transforms.Grayscale(),              # Convert to grayscale if model expects 1 channel
    transforms.ToTensor(),               
    transforms.Normalize((0.5,), (0.5,)) # Normalize for grayscale
])

# Define class names (update according to your dataset)
class_names = ["class0", "class1", "class2", "class3", "class4", 
               "class5", "class6", "class7", "class8", "class9"]

# Start camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Preprocess the frame
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
        label = class_names[predicted.item()]

    # Display prediction
    cv2.putText(frame, f"Prediction: {label}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
