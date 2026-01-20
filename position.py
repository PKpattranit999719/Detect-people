import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# อ่านข้อมูลจาก CSV
density_file_path = "C:/PROJECT/results/myModel/density_output.csv"
df = pd.read_csv(density_file_path)

# แปลงข้อมูลเป็น NumPy array (ขนาด 10x10)
density_array = df.iloc[:, 1:].values  # เอาคอลัมน์จาก Col1 ถึง Col10

# ตรวจสอบขนาดของ array
print(density_array.shape)  # ควรจะได้ (10, 10)

# อ่านภาพห้องเรียน
image_path = "C:/PROJECT/classroom.jpeg"
img = Image.open(image_path)
img_width, img_height = img.size

# กำหนดขนาดกริด (10x10)
grid_size = 10
cell_width = img_width // grid_size
cell_height = img_height // grid_size

# พล็อตภาพ
fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100))
ax.imshow(img, extent=[0, img_width, img_height, 0])

# พล็อตตารางกริด (เพื่อแบ่งแยกภาพ)
ax.grid(True, which='both', axis='both', linestyle='--', color='white', alpha=0.5)
ax.set_xticks(np.arange(0, img_width, cell_width))
ax.set_yticks(np.arange(0, img_height, cell_height))

# พล็อตจุดตามค่าความหนาแน่น โดยปรับขนาดของจุดตามค่าความหนาแน่น
for row in range(grid_size):
    for col in range(grid_size):
        if density_array[row, col] > 0:  # กรองเฉพาะค่าที่ไม่ใช่ศูนย์
            x_pos = col * cell_width + cell_width / 2
            y_pos = row * cell_height + cell_height / 2
            
            # ปรับขนาดจุดตามค่าความหนาแน่น
            point_size = 500 * density_array[row, col]  # ปรับขนาดจุดตามค่าความหนาแน่น
            ax.scatter(x_pos, y_pos, color='red', s=point_size, alpha=0.6)

# ตั้งค่าขอบเขตของกราฟ
ax.set_xlim(0, img_width)
ax.set_ylim(img_height, 0)  # กลับแกน Y

# เพิ่มคำอธิบาย
plt.title("Person Density in Classroom")
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")

# 🔹 กำหนด path สำหรับเซฟภาพ
output_dir = "C:/PROJECT/results/myModel"
output_path = os.path.join(output_dir, "classroom_with_density_and_grid.png")

# บันทึกแผนภาพเป็นไฟล์ PNG
plt.savefig(output_path, bbox_inches='tight')

# แสดงกราฟ
plt.show()