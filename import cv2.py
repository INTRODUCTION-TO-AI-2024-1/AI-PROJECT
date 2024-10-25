import cv2
# Đọc một frame để xử lý (ví dụ frame đã trích xuất)
frame = cv2.imread('1.jpg')

# Chuyển đổi sang ảnh xám
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Lưu ảnh xám
cv2.imwrite('gray_frame.jpg', gray_frame)
import numpy as np

# Đọc ảnh xám (sau khi đã chuyển đổi từ bước trước)
gray_frame = cv2.imread('gray_frame.jpg', 0)

# Áp dụng bộ lọc Gaussian để giảm nhiễu
blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

# Tạo kernel (nhân lọc) cho các phép toán hình thái học
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

# Phép toán Top Hat (làm nổi các chi tiết sáng)
top_hat = cv2.morphologyEx(blurred_frame, cv2.MORPH_TOPHAT, kernel)

# Phép toán Black Hat (làm nổi các chi tiết tối)
black_hat = cv2.morphologyEx(blurred_frame, cv2.MORPH_BLACKHAT, kernel)

# Cộng và trừ các kết quả với ảnh gốc để tăng độ tương phản
enhanced_frame = cv2.add(gray_frame, top_hat)  # Tăng độ tương phản các vùng sáng
enhanced_frame = cv2.subtract(enhanced_frame, black_hat)  # Tăng độ tương phản các vùng tối

# Lưu kết quả cuối cùng sau khi xử lý
cv2.imwrite('enhanced_frame.jpg', enhanced_frame)

# Hiển thị hình ảnh sau khi xử lý
cv2.imshow('Original Frame', gray_frame)
cv2.imshow('Enhanced Frame', enhanced_frame)

# Đợi phím bất kỳ để đóng cửa sổ
cv2.waitKey(0)
cv2.destroyAllWindows()
