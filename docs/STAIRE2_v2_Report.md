# BÁO CÁO CẢI TIẾN GIAI ĐOẠN 2 — ĐỢT 2 (STAIR-Enhanced v2a Bounded)
## Module 1 Refined: Residual-Whitening Projector với Magnitude Matching & Bounded Sigmoid

**Phiên bản:** STAIR-Enhanced v2a_bounded  
**Ngày:** 2026-08-25  
**Tác giả:** KLTN HCMUS — Lê Hà Thanh Chương, Bùi Trung Hiếu  
**Trạng thái:** ✅ Đã hoàn tất cài đặt — Đang chạy thực nghiệm Kaggle

---

## 1. BỐI CẢNH VÀ PHÂN TÍCH LỖI V2A BAN ĐẦU

Tại phiên bản **v2a (Residual Dynamic)** đầu tiên, nhóm đã phát hiện hiện tượng mô hình không thể hội tụ tốt và điểm số Recall/NDCG vẫn thua kém STAIR Baseline. Phân tích log huấn luyện (thông qua `logs/v2fix.md`) đã chỉ ra nguyên nhân gốc rễ là sự **"Bùng nổ nhánh thặng dư (Residual Explosion)"**.

**Chi tiết nguyên nhân:**
1. **Unbounded `lambda_res`:** Tham số $\lambda_{res}$ được khởi tạo là $0.1$ nhưng không có cơ chế chặn biên (unbounded). Khi huấn luyện với **BPR Loss** (hàm loss dạng pairwise margin), mô hình có xu hướng tham lam (greedy) đẩy giá trị $\lambda_{res}$ lên rất cao (đạt tới $> 1.0$) để nhanh chóng phân tách Positive/Negative items, dẫn đến việc nhánh thặng dư $\Delta$ hoàn toàn lấn át nhánh Structural Prior SVD.
2. **Mismatch Magnitude:** Nhánh $\mathbf{E}_{svd}$ được scale bằng hệ số $\sqrt{N_I / D}$, trong khi đó nhánh $\Delta$ đầu ra chỉ là vector L2-normalized (magnitude $= 1$). Sự chênh lệch độ lớn này gây bất ổn khi cộng hai nhánh lại với nhau.

---

## 2. CHI TIẾT KIẾN TRÚC CẢI TIẾN: V2A BOUNDED

Để khắc phục triệt để vấn đề trên và buộc mô hình phải tôn trọng không gian SVD (decorrelated space), 4 chốt chặn an toàn (safeguards) đã được thiết kế và tích hợp:

### 2.1 Chuẩn hóa Scale (Magnitude Matching)
Vector thặng dư $\Delta$ giờ đây được chủ động "bơm" (scale) cho bằng đúng độ lớn của không gian SVD trước khi cộng.
$$scale = \sqrt{\frac{N_{I}}{D}}$$
$$\Delta_{scaled} = \Delta \times scale$$

### 2.2 Bounded Sigmoid (Chặn biên độ Residual)
Thay vì để gradient tự do đẩy $\lambda$, biến $\lambda_{res}$ được thay thế bằng biến thô $\lambda_{raw}$. Giá trị thực tế tham gia vào computational graph bị ép qua hàm Sigmoid và nhân với ngưỡng tối đa $0.3$:
$$\lambda_{actual} = \max\_\lambda \times \sigma(\lambda_{raw})$$
- Biến đổi này đảm bảo nhánh thặng dư $\Delta$ **luôn luôn** chỉ đóng vai trò phụ (đóng góp tối đa $30\%$ lượng thông tin) so với không gian Structural Prior.

### 2.3 Phân tách Optimizer Groups (Tối ưu hóa học tham số)
Hệ số $\lambda_{raw}$ được cấp quyền kiểm soát học riêng biệt trong bộ tối ưu `AdamWSEvo`:
- Áp dụng `weight_decay = 0.1` để tạo lực kéo (regularization) ép $\lambda_{raw}$ lùi về 0 nếu nhánh thặng dư không thực sự mang lại lợi ích giảm loss.
- `lr_lambda = lr_proj \times 0.5`: Cập nhật chậm lại để tránh dao động (oscillation).

### 2.4 Warm-up Training (Đóng băng gradient 50 epoch)
Mô hình cần thời gian để hệ thống User/Item IDs tìm đường đi đúng trong không gian SVD khởi tạo. Do đó, trong **50 epoch đầu tiên**:
- Gradient của $\lambda_{raw}$ bị can thiệp và gán bằng 0 (`zero_grad()`).
- Nhánh thặng dư bị "đóng băng" cứng ở mức $\lambda_{actual} \approx 15\%$ (tương đương với $\lambda_{raw} = 0.0$).

---

## 3. THAY ĐỔI CỤ THỂ TRONG MÃ NGUỒN

Các file mã nguồn quan trọng đã được tái cấu trúc:

1. **[`models/residual_projector_v2.py`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/models/residual_projector_v2.py):**
   - Đổi tên biến `lambda_res` thành `lambda_raw`.
   - Viết lại hàm `composite_embeddings()` để bao gồm logic Magnitude Matching và Sigmoid Bounded.
   - Các Test scripts nội bộ (Sanity Checks) đã được cấu hình lại để kiểm tra gradient backward thành công theo cơ chế Sigmoid.

2. **[`main_enhanced_v2.py`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/main_enhanced_v2.py):**
   - Định nghĩa lại `marked_params()` tách làm 3 nhánh: `text_proj`, `vis_proj`, và `[lambda_raw]`.
   - Chèn logic can thiệp đóng băng gradient ở hàm `train_per_epoch()`:
   ```python
   # Warm-up lambda_raw: dong bang trong 50 epoch dau
   if epoch <= 50:
       if self.model.res_projector.lambda_raw.grad is not None:
           self.model.res_projector.lambda_raw.grad.zero_()
   ```

3. **[`notebook/stair_enhanced_v2a.ipynb`](file:///d:/4thY_HCMUS/KLTN/STAIR-Enhanced/notebook/stair_enhanced_v2a.ipynb) và Generator Script:**
   - Cập nhật hàm xuất Log từ `value=` sang định dạng mới hiển thị đồng thời cả ba giá trị `actual=... | raw=... | grad=...`.
   - Cập nhật Regex Regex parser trong Notebook (`parse_lambda_res_history`) để thu thập chính xác giá trị `actual` nhằm vẽ biểu đồ theo dõi đường cong giới hạn của $\lambda$ thực tế.

---

## 4. BƯỚC TIẾP THEO

1. **Thực thi Kaggle Notebook:** Người dùng đã có thể `git pull` code mới nhất trên Kaggle và chạy lại thực nghiệm trên dataset Baby.
2. **Quan sát Biểu đồ `lambda`:** Mục tiêu là xem biểu đồ $\lambda_{actual}$ bị giam ở mức $0.15$ trong 50 epoch đầu, sau đó từ từ tăng nhẹ nhưng tuyệt đối không vượt quá $0.3$.
3. **Chuyển giao Module 2:** Nếu thí nghiệm `v2a_bounded` cho kết quả tiệm cận hoặc vượt đỉnh Baseline, nhóm sẽ chốt kiến trúc Module 1 và tiếp tục tiến tới triển khai mã nguồn cho **Module 2: Stepwise Graph Contrastive Learning (GCL)**.
