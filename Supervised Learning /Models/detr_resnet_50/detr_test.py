import warnings
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont 
import sys

# 1. Configuration and Setup
warnings.filterwarnings("ignore", category=UserWarning)

# **UPDATE THE FILE PATH TO .png**
local_image_path = sys.argv[1]

# 2. Load Image and Pipeline
print("Loading image and model...")
try:
    # Load the image from the .png file. 
    # .convert("RGB") is essential as most models expect 3 color channels, even if the PNG has an alpha channel.
    original_image = Image.open(local_image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: File not found at {local_image_path}")
    exit()
except Exception as e:
    print(f"An error occurred while opening the image: {e}")
    exit()

# Create the Object Detection pipeline
detector = pipeline("object-detection", model="facebook/detr-resnet-50")

# 3. Run Inference
results = detector(original_image)
print(f"Inference complete. Detected {len(results)} objects.")

# 4. Draw Bounding Boxes and Labels
draw = ImageDraw.Draw(original_image)
line_color = "red"
line_width = 3

# Optional: Load a font for cleaner labels
try:
    font = ImageFont.truetype("arial.ttf", 36)
except IOError:
    font = ImageFont.load_default() 

for detection in results:
    # Extract coordinates, label, and score
    box = detection['box']
    xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
    label = detection['label']
    score = detection['score']

    # Draw the bounding box
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=line_color, width=line_width)

    # Prepare label text and background
    text = f"{label} ({score:.2f})"
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]

    # Draw a solid background rectangle for the text
    draw.rectangle([xmin, ymin - text_height, xmin + text_width, ymin], fill=line_color)

    # Draw the text label
    draw.text((xmin, ymin - text_height), text, fill="white", font=font)


# 5. Display or Save the Image

# **UPDATE THE OUTPUT PATH TO .png**
output_path = f"out.png"
original_image.save(output_path)
print(f"\nImage with detections saved to: {output_path}")
