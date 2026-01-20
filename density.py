import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# อ่านข้อมูลจากไฟล์ CSV ที่มีตำแหน่งของบุคคล
csv_path = "C:/PROJECT/results/myModel/all_student_positions.csv"
df = pd.read_csv(csv_path)

# ตรวจสอบว่า DataFrame ไม่ว่างเปล่า
if not df.empty:
    # กำหนดขนาดกริด (10x10)
    grid_size = 10
    grid = np.zeros((grid_size, grid_size))  # สำหรับเก็บข้อมูลความหนาแน่น

    # คำนวณพิกัดของกริดและเพิ่มจำนวนบุคคลในกริดที่ตำแหน่ง Centroid
    for index, row in df.iterrows():
        centroid_x = row['centroid_x']
        centroid_y = row['centroid_y']
        
        # คำนวณกริดที่ Centroid อยู่
        grid_x = min(int(centroid_x // (1280 // grid_size)), grid_size - 1)  # แบ่งภาพเป็นกริด
        grid_y = min(int(centroid_y // (720 // grid_size)), grid_size - 1)
        
        # เพิ่มจำนวนบุคคลในกริดนั้น
        grid[grid_y, grid_x] += 1

    # คำนวณค่าเฉลี่ยความหนาแน่นในแต่ละกริด
    cell_area = (1280 * 720) / (grid_size ** 2)  # พื้นที่กริด (ขนาดภาพ 1280x720)
    density = grid / cell_area  # ความหนาแน่นในแต่ละกริด

    # สร้าง DataFrame สำหรับบันทึกข้อมูลความหนาแน่น
    density_df = pd.DataFrame(density, columns=[f'Col{i+1}' for i in range(grid_size)])
    density_df.insert(0, 'Row', range(1, grid_size + 1))

    # สร้าง path สำหรับเก็บผลลัพธ์
    result_dir = "C:/PROJECT/results/myModel"
    
    # สร้างชื่อไฟล์ CSV
    density_output = os.path.join(result_dir, "density_output.csv")
    
    # บันทึกข้อมูลความหนาแน่นลงในไฟล์ CSV
    density_df.to_csv(density_output, index=False)

    # สร้างกราฟเส้นสำหรับความหนาแน่น
    plt.figure(figsize=(10, 6))

    # สร้างกราฟเส้นสำหรับแต่ละแถว (Row)
    for row in range(grid_size):
        plt.plot(range(grid_size), density[row, :], label=f'Row {row + 1}')

    # เพิ่มป้ายกำกับและข้อมูลเพิ่มเติม
    plt.xlabel('Zone')
    plt.ylabel('Density (persons/square meter)')
    plt.title('Line Plot of Person Density for Each Row')
    plt.legend(title="Rows")
    plt.grid(True)

    # บันทึกแผนภาพเป็นไฟล์ PNG
    density_plot = "C:/PROJECT/results/myModel/density_plot.png"
    plt.savefig(density_plot, dpi=300, bbox_inches='tight')

    # แสดงกราฟ
    plt.show()

else:
    print("ไม่พบข้อมูลใน CSV")
