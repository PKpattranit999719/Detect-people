import cv2
import numpy as np
import os
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

# โหลดโมเดลที่มีความแม่นยำสูงขึ้น
model = YOLO("model.pt")

# กำหนดเส้นทางโฟลเดอร์ที่มีไฟล์ภาพ
input_folder = "C:/PROJECT/images/0303 08-17/"  # โฟลเดอร์ใหญ่ที่มีโฟลเดอร์ย่อยต่าง ๆ

output_folder = "C:/PROJECT/results/myModel"

# ตรวจสอบและสร้างโฟลเดอร์สำหรับผลลัพธ์ที่ตรวจจับ
detected_images_folder = os.path.join(output_folder, "detected_images")

# ตรวจสอบว่าโฟลเดอร์ที่ต้องการมีอยู่หรือไม่ ถ้ามีให้เพิ่มตัวเลขต่อท้าย
i = 1
base_detected_folder = detected_images_folder
while os.path.exists(detected_images_folder):
    detected_images_folder = f"{base_detected_folder}_{i}"
    i += 1

# สร้างโฟลเดอร์ผลลัพธ์
os.makedirs(output_folder, exist_ok=True)
os.makedirs(detected_images_folder, exist_ok=True)

# กำหนดชื่อไฟล์ CSV สำหรับบันทึกค่าทั้งหมด
csv_path = os.path.join(output_folder, "all_student_positions.csv")

# ตรวจสอบว่ามีไฟล์ CSV อยู่หรือไม่ (ถ้าไม่มี ให้สร้างไฟล์พร้อม Header)
if not os.path.exists(csv_path):
    df_init = pd.DataFrame(columns=["image_file", "x1", "y1", "x2", "y2", "confidence", "centroid_x", "centroid_y"])
    df_init.to_csv(csv_path, index=False)

# ดึงรายการไฟล์ทั้งหมดในโฟลเดอร์ย่อยด้วย os.walk()
image_files = []
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.endswith(('.jpeg', '.jpg', '.png')):
            image_files.append(os.path.join(root, file))

# กำหนดขนาดกริด
grid_size = 10  # ขนาดกริด (10x10)
grid = np.zeros((grid_size, grid_size))  # สำหรับเก็บข้อมูลความหนาแน่น

# วนลูปตรวจจับภาพในทุกไฟล์ในโฟลเดอร์
all_data = []
for image_path in image_files:
    image_file = os.path.basename(image_path)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ ไม่พบไฟล์ภาพ: {image_file}")
        continue

    # ปรับ Contrast และ Brightness
    alpha, beta = 1.2, 20
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # ลด Noise
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # ใช้ CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # ปรับขนาดภาพ
    image_resized = cv2.resize(image, (1280, 720))

    # ตรวจจับบุคคล
    results = model(image_resized, conf=0.3, iou=0.2, classes=[0], max_det=2000)

    # เก็บข้อมูลตำแหน่งของบุคคล
    data = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            if confidence > 0.5:
                centroid_x = (x1 + x2) / 2
                centroid_y = (y1 + y2) / 2
                data.append([image_file, x1, y1, x2, y2, confidence, centroid_x, centroid_y])

                cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(image_resized, (int(centroid_x), int(centroid_y)), 5, (0, 0, 255), -1)

                grid_x = min(int(centroid_x // (image.shape[1] // grid_size)), grid_size - 1)
                grid_y = min(int(centroid_y // (image.shape[0] // grid_size)), grid_size - 1)
                grid[grid_y, grid_x] += 1

    if data:
        df = pd.DataFrame(data, columns=["image_file", "x1", "y1", "x2", "y2", "confidence", "centroid_x", "centroid_y"])
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"✅ บันทึกข้อมูลลง CSV: {csv_path}")

    output_image_path = os.path.join(detected_images_folder, f"output_{image_file}")
    cv2.imwrite(output_image_path, image_resized)
    print(f"✅ บันทึกภาพผลลัพธ์: {output_image_path}")

max_density = np.unravel_index(np.argmax(grid), grid.shape)
print(f"กริดที่มีความหนาแน่นมากที่สุด: {max_density}, จำนวนบุคคล: {grid[max_density]}")


# คำนวณค่า Centroid รวมทั้งหมดจากไฟล์ CSV
df = pd.read_csv(csv_path)
if not df.empty:
    total_centroid_x = df['centroid_x'].mean()
    total_centroid_y = df['centroid_y'].mean()
    print(f"ค่า Centroid รวมทั้งหมด: X = {total_centroid_x:.2f}, Y = {total_centroid_y:.2f}")
else:
    print("ไม่พบข้อมูล Centroid ใน CSV")

print("✅ ตรวจจับภาพทั้งหมดเสร็จสิ้น!")
