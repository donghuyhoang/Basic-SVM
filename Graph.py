import numpy as np
import matplotlib.pyplot as plt
import math  # Import thư viện math để sử dụng sqrt

# Đọc dữ liệu từ file
try:
    data = np.loadtxt("data.txt", delimiter=",")
    X = data[:, :2]
    y = data[:, 2]
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file data.txt")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc data.txt: {e}")
    exit()

# Đọc siêu phẳng từ file
try:
    with open("hyperplane.txt", "r") as f:
        line = f.readline().strip()
        w1, w2, b = map(float, line.split(","))
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file hyperplane.txt")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc hyperplane.txt: {e}")
    exit()

# Đọc margin từ file (Mặc dù không dùng trực tiếp để tính offset nữa,
# nhưng vẫn có thể giữ lại để kiểm tra hoặc mục đích khác nếu muốn)
try:
    with open("margin.txt", "r") as f:
        margin_value_from_file = float(f.readline().strip()) 
        # margin_value_from_file là 2.0 / ||w|| từ file C
        print(f"Giá trị margin (tổng độ rộng 2/||w||) đọc từ file: {margin_value_from_file:.4f}")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file margin.txt")
    # Có thể không cần exit() nếu bạn không dùng giá trị này nữa
    margin_value_from_file = None 
except Exception as e:
    print(f"Lỗi khi đọc margin.txt: {e}")
    margin_value_from_file = None

# Vẽ dữ liệu
plt.figure(figsize=(10, 8)) 
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Lớp 1', s=50, edgecolors='k', alpha=0.7)
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Lớp -1', s=50, edgecolors='k', alpha=0.7)


# Tính toán cho đường siêu phẳng và lề
x_min_data, x_max_data = X[:, 0].min(), X[:, 0].max()
y_min_data, y_max_data = X[:, 1].min(), X[:, 1].max()

# Mở rộng phạm vi một chút để đường vẽ không bị cắt cụt
plot_margin_x = (x_max_data - x_min_data) * 0.1
plot_margin_y = (y_max_data - y_min_data) * 0.1

x_vals = np.linspace(x_min_data - plot_margin_x, x_max_data + plot_margin_x, 200)


# Kiểm tra w2 để tránh lỗi chia cho 0
if abs(w2) < 1e-6: 
    if abs(w1) < 1e-6: 
        print("Lỗi: w1 và w2 không thể đồng thời bằng 0.")
    else:
        decision_boundary_x = -b / w1
        plt.axvline(x=decision_boundary_x, color='k', linestyle='-', label="Đường biên Quyết định")
        
        # Đối với SVM, các đường lề tương ứng với w.x + b = +1 và w.x + b = -1
        # Do đó, x_margin_plus = (-b + 1) / w1
        # và x_margin_minus = (-b - 1) / w1
        plt.axvline(x=(-b + 1.0) / w1, color='gray', linestyle='--', label="Lề")
        plt.axvline(x=(-b - 1.0) / w1, color='gray', linestyle='--')
else:
    y_vals_decision = (-w1 * x_vals - b) / w2
    plt.plot(x_vals, y_vals_decision, 'k-', label="Đường biên Quyết định")

    # Giá trị dịch chuyển cho các đường lề chuẩn của SVM là 1.0
    # (tương ứng với w.x + b = +1 và w.x + b = -1)
    margin_offset_val = 1.0

    y_vals_margin_plus = (-w1 * x_vals - b + margin_offset_val) / w2
    y_vals_margin_minus = (-w1 * x_vals - b - margin_offset_val) / w2

    plt.plot(x_vals, y_vals_margin_plus, 'k--', color='gray', label="Lề") 
    plt.plot(x_vals, y_vals_margin_minus, 'k--', color='gray')

# Tính toán và hiển thị lề hình học thực tế từ w1, w2
norm_w_python = math.sqrt(w1**2 + w2**2)
if norm_w_python > 1e-9:
    geometric_margin_one_side = 2.0 / norm_w_python
    print(f"Lề hình học tính toán (2/||w||) từ w1,w2 trong Python: {geometric_margin_one_side:.4f}")
    # Nếu muốn, có thể so sánh với margin_value_from_file / 2.0
    if margin_value_from_file is not None:
        print(f"So sánh với (margin từ file C / 2): {margin_value_from_file / 2.0:.4f}")


plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("SVM - Đường biên Quyết định và Lề")
plt.grid(True, linestyle=':', alpha=0.5)
plt.axhline(0, color='black', linewidth=0.5, alpha=0.5) 
plt.axvline(0, color='black', linewidth=0.5, alpha=0.5) 

# Đặt giới hạn trục dựa trên dữ liệu và một chút lề
plt.xlim(x_min_data - plot_margin_x - 0.1, x_max_data + plot_margin_x + 0.1) 
plt.ylim(y_min_data - plot_margin_y - 0.1, y_max_data + plot_margin_y + 0.1)
plt.gca().set_aspect('equal', adjustable='box') 
plt.show()
