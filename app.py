import os
import io
from PIL import Image

from flask import Flask, request, jsonify
from torchvision import transforms
import torch
from model_class import Model

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model().to(device)
model.load_state_dict(torch.load("./models/emotion_detection_model.pt", map_location=torch.device(device)))
model.eval()


def transform_image(img):
    """
    :param img: PIL image
    :return:
    """
    transform = transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])

    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)

    return img


def predict(img):
    _, prediction = torch.max(model(img), dim=1)
    return model.classes[prediction.item()]


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def call_model():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes))
            tensor = transform_image(pillow_img)
            prediction = predict(tensor)
            data = {"prediction": prediction}

            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)
