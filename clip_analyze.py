import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Define descriptive phrases
basic_objects = ["dog", "car", "tree", "person", "bicycle"]
interactions = ["playing fetch", "driving", "sitting under a tree", "riding a bicycle"]
thematic_contexts = ["leisure time in a park", "morning commute"]

# Initialize the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def analyze_image(image_path):
    image = Image.open(image_path)
    inputs = processor(text=basic_objects + interactions + thematic_contexts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image.softmax(dim=1).tolist()[0]

    results = {}
    for i, phrase in enumerate(basic_objects + interactions + thematic_contexts):
        results[phrase] = logits_per_image[i]

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    top_phrases = [item[0] for item in sorted_results[:5]]

    summary = f"This scene likely involves {', '.join(top_phrases)}."

    return {
        "image_name": os.path.basename(image_path),
        "inferred_activities": sorted_results,
        "summary": summary
    }

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".jpeg", ".png", ".webp")):
            try:
                image_path = os.path.join(directory, filename)
                analysis = analyze_image(image_path)

                json_path = os.path.join(directory, f"{os.path.splitext(filename)[0]}.json")
                with open(json_path, 'w') as json_file:
                    json.dump(analysis, json_file, indent=4)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
    else:
        process_directory(sys.argv[1])
