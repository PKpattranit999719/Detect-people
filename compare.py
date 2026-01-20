import cv2
import numpy as np
import matplotlib.pyplot as plt

# 🔹 กำหนดพาธของไฟล์
image_path = "C:/PROJECT/classroom.jpeg"  
heatmap_path = "C:/PROJECT/results/myModel/heatmap.png"  

# โหลดภาพต้นฉบับ
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# โหลด Heatmap
heatmap = cv2.imread(heatmap_path)

# 🔥 แปลงเป็นขาวดำเพื่อตัดขอบ
heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

# ใช้ Adaptive Threshold เพื่อลบขอบขาว
thresh = cv2.adaptiveThreshold(heatmap_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2) 

# 🔍 หาขอบเขตของ Heatmap จริง
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)  # เลือก Contour ใหญ่สุด

if contours:
    x, y, w, h = cv2.boundingRect(contours[0])  # ใช้ Bounding Box ใหญ่สุด
    heatmap_cropped = heatmap[y:y+h, x:x+w]
else:
    heatmap_cropped = heatmap  # ถ้าไม่เจอ Contour ให้ใช้ภาพเดิม

# ✅ แปลงเป็น RGB เพื่อป้องกันสีเพี้ยน
heatmap_cropped = cv2.cvtColor(heatmap_cropped, cv2.COLOR_BGR2RGB)

# ✅ Normalize ค่า Heatmap เพื่อป้องกันสีผิดเพี้ยน
heatmap_cropped = cv2.normalize(heatmap_cropped, None, 0, 255, cv2.NORM_MINMAX)
heatmap_cropped = heatmap_cropped.astype(np.uint8)

# ปรับขนาดให้ตรงกับภาพห้องเรียน
img_height, img_width, _ = image.shape
heatmap_resized = cv2.resize(heatmap_cropped, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

# 🔹 ซ้อน Heatmap บนภาพต้นฉบับ
overlay = cv2.addWeighted(image, 0.7, heatmap_resized, 0.5, 0)

# 🔹 บันทึกผลลัพธ์ overlay เป็น .png
output_path = "C:/PROJECT/results/myModel/compare_output.png"
overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)  # แปลงกลับเป็น BGR สำหรับเซฟ
cv2.imwrite(output_path, overlay_bgr)

# แสดงผล
plt.figure(figsize=(10, 6))
plt.imshow(overlay)
plt.axis("off")
plt.title("Overlay of Cropped Heatmap on Classroom Image")
plt.show()
