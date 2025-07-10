from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from deep_translator import GoogleTranslator

# Generate caption dari gambar
def generate_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption_en = processor.decode(out[0], skip_special_tokens=True)
    return caption_en

# Translate caption ke Bahasa Indonesia
def translate_caption(text):
    return GoogleTranslator(source="en", target="id").translate(text)
