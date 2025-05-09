import json

def load_fertilizer_data(filepath="fertilizer_data/recommendations.json"):
    with open(filepath, "r") as f:
        return json.load(f)

def get_fertilizer_info(disease_name, fertilizer_data):
    info = fertilizer_data.get(disease_name)
    if info:
        return f"💡 Fertilizer: {info['fertilizer']}\n🧪 Dosage: {info['dosage']}"
    return "❗ No fertilizer info available for this disease."
