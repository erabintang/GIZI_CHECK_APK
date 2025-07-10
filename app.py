from flask import Flask, render_template, request
from model_replicate import detect_food_yolo
from dotenv import load_dotenv
import os
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
from face_detector import detect_face_and_analyze
from deepface import DeepFace

# === Inisialisasi ===
load_dotenv()
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Load Data Gizi ===
with open('nutrition_db.json') as f:
    nutrition_data_file = json.load(f)

nutrition_data = {
    "ayam goreng": {"kalori": 250, "lemak": 15, "kategori": "tidak sehat"},
    "nasi putih": {"kalori": 200, "lemak": 1, "kategori": "netral"},
    "sayur bayam": {"kalori": 50, "lemak": 0.5, "kategori": "sehat"},
    "es teh manis": {"kalori": 120, "lemak": 0, "kategori": "tidak sehat"},
    "tempe goreng": {"kalori": 190, "lemak": 10, "kategori": "sehat"},
    "ikan bakar": {"kalori": 180, "lemak": 8, "kategori": "sehat"},
}

# === Caption AI ===
def generate_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption_en = processor.decode(out[0], skip_special_tokens=True)
    return caption_en

def translate_caption(text):
    return GoogleTranslator(source="en", target="id").translate(text)

# === Deteksi Gambar ===
def detect_food_from_image(image_path):
    detected_labels = detect_food_yolo(image_path)
    cocok = []
    for label in detected_labels:
        nama = label.lower()
        if nama in nutrition_data:
            cocok.append(nama)
    return cocok  


# === ROUTES ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    file = request.files['food_image']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    caption_en = generate_caption(filepath)
    caption_id = translate_caption(caption_en)

    detected = detect_food_from_image(filepath)
    total_kalori = 0
    status_score = 0
    results = []

    for food in detected:
        info = nutrition_data.get(food, {})
        kal = info.get("kalori", 0)
        lemak = info.get("lemak", 0)
        kategori = info.get("kategori", "?")
        total_kalori += kal
        if kategori == "sehat":
            status_score += 1
        elif kategori == "tidak sehat":
            status_score -= 1
        results.append({"nama": food, "kalori": kal, "lemak": lemak, "kategori": kategori})

    if status_score > 0:
        final_status = "Makanan ini cenderung sehat"
    elif status_score < 0:
        final_status = "Makanan ini cenderung tidak sehat"
    else:
        final_status = "Makanan ini netral"

    return render_template("result.html",
        image_path=filepath,
        caption=caption_id,
        results=results,
        total_calories=total_kalori,
        final_status=final_status
    )

# === BMI ===
@app.route('/bmi', methods=['GET', 'POST'])
def bmi():
    if request.method == 'POST':
        berat = float(request.form['berat'])
        tinggi = float(request.form['tinggi']) / 100
        if tinggi <= 0:
            return "Tinggi tidak valid"

        bmi_value = berat / (tinggi ** 2)
        if bmi_value < 18.5:
            status = "Kurus"
        elif 18.5 <= bmi_value < 25:
            status = "Normal"
        elif 25 <= bmi_value < 30:
            status = "Gemuk"
        else:
            status = "Obesitas"

        return render_template('bmi_result.html', bmi=round(bmi_value, 2), status=status)
    return render_template('bmi_form.html')

# === Kuis ===
def ai_komentar(aktivitas):
    aktivitas = aktivitas.lower()
    if any(kata in aktivitas for kata in ["olahraga", "lari", "senam", "yoga"]):
        return "Bagus! Kamu aktif bergerak di waktu ini. Olahraga sangat penting untuk metabolisme. 💪"
    elif any(kata in aktivitas for kata in ["scroll", "main hp", "rebahan", "tidur lagi"]):
        return "Hmm... coba lebih aktif ya di waktu ini. Terlalu pasif bisa bikin tubuh kurang bertenaga. 📱➡️🚶"
    elif any(kata in aktivitas for kata in ["belajar", "kerja", "ngoding", "sekolah"]):
        return "Aktivitasmu produktif sekali. Pastikan tetap jaga keseimbangan dengan istirahat cukup. 📚"
    elif any(kata in aktivitas for kata in ["makan", "sarapan"]):
        return "Kamu sudah sarapan, itu bagus. Energi pagi sangat penting untuk memulai hari. 🍽️"
    elif any(kata in aktivitas for kata in ["tidur", "istirahat"]):
        return "Istirahat yang cukup penting untuk pemulihan tubuh. Jangan terlalu begadang ya! 😴"
    else:
        return "Aktivitas ini cukup unik! Terus jaga gaya hidup sehatmu. 👍"

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    if request.method == "POST":
        pagi = request.form.get("pagi", "")
        siang = request.form.get("siang", "")
        malam = request.form.get("malam", "")

        komentar_pagi = ai_komentar(pagi)
        komentar_siang = ai_komentar(siang)
        komentar_malam = ai_komentar(malam)

        return render_template("quiz_result.html",
            pagi=pagi,
            siang=siang,
            malam=malam,
            komentar_pagi=komentar_pagi,
            komentar_siang=komentar_siang,
            komentar_malam=komentar_malam
        )

    return render_template("quiz.html")

@app.route('/facecheck', methods=['POST'])
def face_check():
    img = request.files['image']
    # Pindahkan proses DeepFace di sini
    result = DeepFace.analyze(img_path = img, actions = ['age', 'gender'])
    return jsonify(result)

@app.route("/face-logs")
def face_logs():
    files = os.listdir("static/face_logs")
    images = [f"/static/face_logs/{file}" for file in sorted(files, reverse=True)]
    return render_template("face_logs.html", images=images)

# === Run ===
if __name__ == '__main__':
    app.run(debug=True)
