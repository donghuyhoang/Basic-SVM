import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import math # Mặc dù không dùng trực tiếp trong phần margin nhưng vẫn giữ nếu cần

def load_data_2d_constrained(file_path):
    """
    Tải dữ liệu 2D từ tệp văn bản với các ràng buộc:
    - Lớp là 1 hoặc -1.
    - Giá trị đặc trưng (x, y) từ 0 đến 1.
    Mỗi dòng có dạng: x,y,class
    Trả về:
        X (numpy array): Mảng các đặc trưng (N_samples, 2).
        y (numpy array): Mảng các nhãn lớp (N_samples,).
    """
    data_features = []
    data_labels = []
    line_num = 0

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line_num += 1
                parts = line.strip().split(',')
                
                # Dữ liệu 2D phải có đúng 3 phần: x, y, class
                if len(parts) != 3:
                    print(f"Cảnh báo dòng {line_num}: Dòng không đúng 3 phần tử (x,y,class). Bỏ qua: '{line.strip()}'")
                    continue

                try:
                    # Lấy đặc trưng và nhãn
                    feature_x_str, feature_y_str, label_str = parts

                    # Chuyển đổi và kiểm tra nhãn
                    label = int(label_str)
                    if label not in [-1, 1]:
                        print(f"Cảnh báo dòng {line_num}: Nhãn '{label_str}' không hợp lệ (phải là 1 hoặc -1). Bỏ qua dòng.")
                        continue

                    # Chuyển đổi và kiểm tra đặc trưng
                    feature_x = float(feature_x_str)
                    feature_y = float(feature_y_str)

                    valid_features = True
                    for i, val in enumerate([feature_x, feature_y]):
                        if not (0 <= val <= 1):
                            print(f"Cảnh báo dòng {line_num}: Giá trị đặc trưng thứ {i+1} '{val}' nằm ngoài khoảng [0,1]. Bỏ qua dòng.")
                            valid_features = False
                            break
                    if not valid_features:
                        continue
                    
                    data_features.append([feature_x, feature_y])
                    data_labels.append(label)

                except ValueError:
                    print(f"Cảnh báo dòng {line_num}: Lỗi chuyển đổi giá trị trong dòng: '{line.strip()}'. Bỏ qua dòng.")
                    continue
        
        if not data_features:
            print("Lỗi: Không có dữ liệu hợp lệ nào được tải từ tệp.")
            return None, None

        X = np.array(data_features)
        y = np.array(data_labels)
        
        print(f"Đã tải {len(X)} điểm dữ liệu 2D hợp lệ.")
        print(f"Các lớp được tìm thấy: {np.unique(y)}")

        return X, y

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp '{file_path}'")
        return None, None
    except Exception as e:
        print(f"Lỗi không xác định khi tải dữ liệu: {e}")
        return None, None

def plot_svm_2d_constrained(X, y, model):
    """Vẽ dữ liệu 2D (0-1) và đường biên quyết định của SVM."""
    plt.figure(figsize=(8, 8))
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=plt.cm.coolwarm, edgecolors='k', vmin=-1, vmax=1)
    
    handles = []
    # model.classes_ sẽ là mảng các lớp duy nhất mà mô hình đã thấy, ví dụ [-1, 1]
    # Đảm bảo rằng chúng ta tạo legend cho các lớp có trong dữ liệu
    # và màu sắc tương ứng với cách cmap ánh xạ giá trị -1 và 1
    if -1 in model.classes_:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=plt.cm.coolwarm(0.0), # Màu cho -1 (thường là đầu của cmap)
                                  label='Lớp -1', markersize=10))
    if 1 in model.classes_:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=plt.cm.coolwarm(1.0), # Màu cho 1 (thường là cuối của cmap)
                                  label='Lớp 1', markersize=10))
    
    if handles: # Chỉ hiển thị legend nếu có handles được tạo
        plt.legend(handles=handles, title="Lớp")

    ax = plt.gca()
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    xx, yy = np.meshgrid(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100),
                         np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100))
    
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # levels=[-1, 0, 1] vẽ đường biên quyết định (0) và hai đường lề (-1 và 1)
    # dựa trên giá trị của decision_function
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.7,
               linestyles=['--', '-', '--'])

    if hasattr(model, "support_vectors_") and model.support_vectors_ is not None and len(model.support_vectors_) > 0 :
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=150,
                   linewidth=1.5, facecolors='none', edgecolors='k', marker='o', label='Vector Hỗ Trợ')
        
        # Cập nhật legend để bao gồm Vector Hỗ Trợ nếu chưa có
        current_handles, current_labels = ax.get_legend_handles_labels()
        # Kiểm tra xem 'Vector Hỗ Trợ' đã có trong legend chưa để tránh trùng lặp
        has_sv_label = any('Vector Hỗ Trợ' in label for label in current_labels)
        
        if not has_sv_label:
            sv_handle = plt.Line2D([0], [0], marker='o', color='w', markersize=10,
                                   markeredgecolor='k', markerfacecolor='none', 
                                   linestyle='None', label='Vector Hỗ Trợ')
            ax.legend(handles=current_handles + [sv_handle], labels=current_labels + ['Vector Hỗ Trợ'])
        elif not handles and has_sv_label : # Nếu ban đầu không có handles (không có lớp) nhưng có SV
             ax.legend()


    plt.xlabel("Đặc trưng X1 (0 đến 1)")
    plt.ylabel("Đặc trưng X2 (0 đến 1)")
    plt.title("SVM với Kernel Tuyến tính (2D, Dữ liệu [0,1], Lớp {-1,1})")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == "__main__":
    file_path = 'data.txt'
    X, y = load_data_2d_constrained(file_path)

    if X is not None and y is not None:
        if len(np.unique(y)) < 2:
            print("Lỗi: Cần ít nhất 2 lớp (cả 1 và -1) để thực hiện phân loại SVM.")
        elif X.shape[1] != 2:
            print(f"Lỗi: Dữ liệu không phải là 2 chiều (số đặc trưng là {X.shape[1]}).")
        else:
            model = svm.SVC(kernel='linear')
            model.fit(X, y)
            print("Đã huấn luyện xong mô hình SVM cho dữ liệu 2D.")
            print(f"  Các lớp trong mô hình: {model.classes_}")
            
            w = model.coef_[0] # Lấy vector trọng số w (là một mảng 1D cho trường hợp 2D)
            b = model.intercept_[0] # Lấy intercept b
            
            print(f"  Hệ số (w): {w}")
            print(f"  Intercept (b): {b}")

            # Tính toán norm của vector w
            norm_w = np.linalg.norm(w)
            
            if norm_w == 0: # Hiếm khi xảy ra nếu mô hình huấn luyện thành công
                print("  Lỗi: Norm của vector trọng số w bằng 0, không thể tính margin.")
            else:
                geometric_margin = 2 / norm_w
                # Độ rộng của toàn bộ vùng lề là 2 * geometric_margin
                # Khoảng cách từ siêu phẳng quyết định đến mỗi đường lề là geometric_margin
                print(f"  Lề hình học (2 / ||w||): {geometric_margin:.4f}") 
                # Các đường lề được vẽ bởi ax.contour với levels=[-1, 0, 1]
                # tương ứng với decision_function(x) = -1, 0, 1.
                # Khoảng cách giữa decision_function(x)=0 và decision_function(x)=1 (hoặc -1)
                # chính là lề hình học này.

            if hasattr(model, "support_vectors_") and model.support_vectors_ is not None:
                 print(f"  Số vector hỗ trợ: {len(model.support_vectors_)}")

            plot_svm_2d_constrained(X, y, model)
    else:
        print("Không thể tiếp tục do lỗi tải dữ liệu hoặc không có dữ liệu hợp lệ.")
