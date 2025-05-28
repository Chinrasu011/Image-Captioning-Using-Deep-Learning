from flask import Flask, request, render_template, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

app = Flask(__name__)

# Load the Processor and Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/caption", methods=["POST"])
def caption_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Load the uploaded image
    image_file = request.files["image"]
    raw_image = Image.open(image_file).convert("RGB")

    # Prepare the inputs
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")

    # Generate the caption
    with torch.no_grad():
        output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return jsonify({"caption": caption})

if __name__ == "__main__":
    app.run(debug=True)