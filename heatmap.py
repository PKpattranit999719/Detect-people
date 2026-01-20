import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# โหลดข้อมูลจาก CSV
csv_path = "C:/PROJECT/results/myModel/all_student_positions.csv"
df = pd.read_csv(csv_path)

# กำหนดขนาดกริด (Grid size ตามที่ต้องการ)
grid_size = 10
heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

# นำข้อมูล Centroid จาก CSV มาใช้
for _, row in df.iterrows():
    centroid_x = row['centroid_x']
    centroid_y = row['centroid_y']
    
    # คำนวณตำแหน่งของ Centroid ในกริด
    grid_x = min(int(centroid_x // (1280 / grid_size)), grid_size - 1)  # 1280 คือความกว้างของภาพ
    grid_y = min(int(centroid_y // (720 / grid_size)), grid_size - 1)   # 720 คือความสูงของภาพ
    
    # เพิ่มค่าในกริดที่ตำแหน่ง Centroid
    heatmap[grid_y, grid_x] += 1

# แสดงผล Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.colorbar()  
plt.title("Heatmap of Student Centroids")

# 🔹 บันทึก Heatmap เป็นไฟล์ PNG
heatmap_path = "C:/PROJECT/results/myModel/heatmap.png"
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')  
plt.close()  # ปิดกราฟเพื่อป้องกันปัญหา

print(f"✅ Heatmap ถูกบันทึกที่: {heatmap_path}")
