import numpy as np
import matplotlib.pyplot as plt
import math  # Import thư viện math để sử dụng sqrt

# Đọc dữ liệu từ file
data = np.loadtxt("data.txt", delimiter=",")
X = data[:, :2]
y = data[:, 2]

# Đọc siêu phẳng từ file
with open("hyperplane.txt", "r") as f:
    line = f.readline().strip()
    w1, w2, b = map(float, line.split(","))

# Đọc support vectors từ file (nếu có)
support_vectors = np.loadtxt("support_vectors.txt", delimiter=",") if "support_vectors.txt" else np.array([])

# Đọc margin từ file
with open("margin.txt", "r") as f:
    margin = float(f.readline().strip())

# Vẽ dữ liệu
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')

# Vẽ vector hỗ trợ (màu vàng)
if support_vectors.size > 0:
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, edgecolors='black', facecolors='yellow', label='Support Vectors')

# Vẽ đường siêu phẳng
x_vals = np.linspace(0, 1, 100)
y_vals = (-w1 * x_vals - b) / w2
plt.plot(x_vals, y_vals, 'k-', label="Decision Boundary")

# Tính vector pháp tuyến chuẩn hóa (normal_vector_normalized[1])
norm = math.sqrt(w1**2 + w2**2)  # Tính độ dài vector pháp tuyến
normal_vector_normalized_1 = w2 / norm  # Tính thành phần thứ hai của vector pháp tuyến chuẩn hóa

# Tạo lề SVM theo hướng vector pháp tuyến
y_vals_pos = y_vals + margin / 2 * normal_vector_normalized_1
y_vals_neg = y_vals - margin / 2 * normal_vector_normalized_1

plt.plot(x_vals, y_vals_pos, 'k--', label="Margin")
plt.plot(x_vals, y_vals_neg, 'k--')

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("SVM Decision Boundary with Support Vectors")
plt.show()
