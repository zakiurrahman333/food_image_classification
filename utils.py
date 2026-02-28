from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_image(model, image, transform, class_names):
    img = Image.open(image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        pred_idx = outputs.argmax(dim=1).item()

    return class_names[pred_idx]