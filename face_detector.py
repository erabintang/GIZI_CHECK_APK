# face_detector.py

from deepface import DeepFace
import cv2
import numpy as np

def detect_face_and_analyze():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Tidak bisa mengakses kamera.")
            break

        try:
            # Analisis wajah dengan DeepFace
            results = DeepFace.analyze(
                frame,
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False
            )

            # Ambil data dari hasil analisis
            dominant_emotion = results[0]["dominant_emotion"]
            age = results[0]["age"]
            gender = results[0]["gender"]
            region = results[0]["region"]

            # Hitung rasio lebar vs tinggi wajah dari bounding box
            w = region["w"]
            h = region["h"]
            ratio = w / h if h != 0 else 0

            # Deteksi kondisi "terlihat gemuk"
            if ratio > 0.9:
                status = "Wajah terlihat bahagia & bulat"
            else:
                status = " Wajah terlihat bugar"

            # Gambar kotak wajah
            x, y = region["x"], region["y"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Tampilkan info
            cv2.putText(frame, f"Emosi: {dominant_emotion}", (x, y - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            cv2.putText(frame, f"Usia: {age} | Gender: {gender}", (x, y - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 200), 2)
            cv2.putText(frame, f"{status}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        except Exception as e:
            cv2.putText(frame, "🔍 Wajah tidak terdeteksi", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            print(f"⚠️ Error analisis: {e}")

        cv2.imshow("GiziCheck AI - DeepFace Analyzer", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:  # q atau ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=False)
