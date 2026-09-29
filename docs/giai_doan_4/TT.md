# KỊCH BẢN THUYẾT TRÌNH BÁO CÁO TIẾN ĐỘ KHÓA LUẬN TỐT NGHIỆP
## HƯỚNG DẪN TRÌNH BÀY TRỰC TIẾP TRÊN BÁO CÁO PDF LATEX (CHƯƠNG 3 - GIAI ĐOẠN 4)

- **Người thực hiện:** Nhóm sinh viên Khóa luận tốt nghiệp
- **Đối tượng báo cáo:** Giảng viên hướng dẫn (thưa cô, xưng em)
- **Hình thức trình bày:** Trình chiếu và cuộn trang trực tiếp trên tệp báo cáo PDF biên dịch từ LaTeX (`03_stair.tex`)
- **Tập dữ liệu đối chuẩn:** Bộ ba tam giác chuẩn Amazon Baby (nhỏ, mật độ cao), Amazon Sports (siêu thưa 99.95%), Amazon Electronics (quy mô công nghiệp 63K items, 1.69M tương tác)
- **Thời lượng dự kiến:** 12 – 15 phút

---

## TỔNG QUAN LỘ TRÌNH VÀ VỊ TRÍ CUỘN TRANG TRÊN FILE BÁO CÁO PDF

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        LỘ TRÌNH TRÌNH BÀY TRÊN BÁO CÁO PDF                              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. MỤC 3.1: STAIR-CNLGCL v1-R (Tích hợp trực giao đa thành phần)                       │
│    • Vị trí: Đầu trang Mục 3.1 ──> Công thức (3.1)-(3.4) ──> Hình 3.1 (TikZ Trực giao) │
│    • Bảng kết quả: Bảng 1 (Metric 3 tập) ──> Bảng 3 (Tiêu thụ tài nguyên VRAM/Thời gian)│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. MỤC 3.2: STAIR-BCCR v2.1 (Điều chuẩn chéo tầng trong không gian phức)               │
│    • Vị trí: Mục 3.2 ──> Công thức chuẩn hóa FSC (3.5) ──> Ánh xạ pha & Givens (3.6)    │
│    • Bảng kết quả: Bảng 4 (Đối chuẩn phục hồi v2 vs v2.1) ──> Bảng 5 (Chi phí O(B²))   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. MỤC 3.3: STAIR-MHD v3 (Multi-Head Hypergraph Disentanglement)                       │
│    • Vị trí: Mục 3.3 ──> Ma trận liên thuộc & Gating (3.7)-(3.8) ──> Hình 3.3 (TikZ)   │
│    • Bảng kết quả: Bảng 8 (Đỉnh cao Sports 0.1129) ──> Bảng 9 (Tài nguyên 8 giờ train) │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. MỤC 3.4: STAIR4-CSGC v4 (Confidence-Shrunk Behavioral Graph Calibration)             │
│    • Vị trí: Mục 3.4 ──> Hệ 8 khối toán học ──> Hình 3.4 (TikZ Hai cột Static vs Online)│
│    • Bảng kết quả: Bảng 11 (Ablation 7 nhánh) ──> Bảng 12 (Tổng hợp) ──> Bảng 13 (Paired)│
│      ──> Phân tích Evidence Scaling ──> Bảng 14 (Zero VRAM 0.00% & Tốc độ gấp 2.2x-4.3x)│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. TỔNG KẾT & PHẢN BIỆN: 3 bài học thiết kế kiến trúc và giải đáp thắc mắc             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## PHẦN 1: MỞ ĐẦU VÀ BỐI CẢNH TỪ BASELINE STAIR
*(Thời lượng: ~1.5 phút)*

### Chỉ dẫn thao tác trên PDF
> Mở đầu Chương 3, dừng ở phần giới thiệu đầu trang trước khi vào Mục 3.1.

### Lời thoại trình bày (Script nói)
> "Kính thưa cô, hôm nay nhóm em xin phép được báo cáo chi tiết về kết quả nghiên cứu và thực nghiệm trong Giai đoạn 4 của đề tài: Các giải pháp cải tiến mô hình STAIR trong bài toán gợi ý đa phương thức.
> 
> Để cô tiện theo dõi trên file báo cáo, em xin tóm lược nhanh xuất phát điểm từ mô hình STAIR gốc được công bố tại hội nghị AAAI 2025. Mô hình baseline STAIR có hai đặc trưng cốt lõi:
> 1. Thứ nhất, cơ chế Forward Stepwise Convolution (FSC) lan truyền thông tin trên đồ thị tương tác người dùng - sản phẩm để tổng hợp biểu diễn nhiều tầng.
> 2. Thứ hai, thuật toán tối ưu hóa AdamWSEvo sử dụng bộ làm mịn Neumann BSC để làm mịn hướng cập nhật gradient của sản phẩm dựa trên đồ thị kNN ngữ nghĩa đa phương thức $S_0$.
> 
> Tuy nhiên, qua quá trình phân tích thực nghiệm, nhóm em phát hiện ra hai vấn đề tồn đọng của baseline:
> - Một là, đồ thị kNN ngữ nghĩa ban đầu được xây dựng hoàn toàn từ vector đặc trưng văn bản và hình ảnh thô. Hai sản phẩm nhìn giống nhau hoặc có mô tả tương tự nhau chưa chắc đã phản ánh cùng một hành vi mua sắm thực tế.
> - Hai là, việc lan truyền biểu diễn cộng tác thuần túy trên đồ thị tương tác thưa thớt dễ làm suy hao các đặc trưng đa phương thức ở các tầng sâu.
> 
> Vì vậy, mục tiêu xuyên suốt của Giai đoạn 4 là tìm kiếm cơ chế kết hợp hiệu quả giữa tín hiệu hành vi người dùng và đặc trưng nội dung đa phương thức, trải qua 4 thế hệ kiến trúc cải tiến liên tục từ v1-R đến v4. Em xin phép được trình bày chi tiết từng phiên bản ngay sau đây ạ."

---

## PHẦN 2: CẢI TIẾN 1 — STAIR-CNLGCL v1-R (TÍCH HỢP TRỰC GIAO ĐA THÀNH PHẦN)
*(Thời lượng: ~2.5 phút)*

### Chỉ dẫn thao tác trên PDF
> Cuộn xuống **Mục 3.1 (trang 1)**, dừng ở đoạn mô tả **Tính trực giao giữa forward pass và backward pass**, trỏ vào **Công thức (1)-(3)**, sau đó lướt qua **Hình 1 (sơ đồ TikZ)** và dừng lại ở **Bảng 1** và **Bảng 3**.

### Lời thoại trình bày (Script nói)
> "Thưa cô, ở phiên bản đầu tiên là **STAIR-CNLGCL v1-R** tại Mục 3.1, nhóm em đặt ra bài toán: làm sao để kết hợp đồng thời việc học biểu diễn đối sánh ở hàm mất mát với việc làm mịn gradient ở bộ tối ưu mà không gây xung đột gradient?
> 
> Như cô có thể thấy ở các công thức từ (1) đến (3) và sơ đồ kiến trúc Hình 1, kiến trúc v1-R được thiết kế dựa trên nguyên lý trực giao hai pha tách biệt:
> - **Trong pha lan truyền xuôi forward pass:** Mô hình tính toán hàm mất mát tổng hợp gồm hàm xếp hạng BPR và hàm đối sánh lân cận hai chiều $\mathcal{L}_{\text{CNLGCL}}$ giữa tầng 0 và tầng 1, có tích hợp mặt nạ lọc mẫu âm giả và điều phối trọng số tăng dần theo lịch trình khởi động warm-up. Cơ chế này chỉ can thiệp vào khoảng cách góc giữa các vector nhúng trên mặt cầu đơn vị.
> - **Trong pha lan truyền ngược backward pass:** Thuật toán AdamWSEvo làm mịn bước cập nhật gradient của sản phẩm bằng ma trận kề tăng cường $\tilde{A}_{\text{boosted}}$. Ma trận này được xây dựng bằng cách kết hợp độ tương đồng cosine đa phương thức với chỉ số đồng mua Ochiai từ ma trận tương tác.
> 
> Do toán tử làm mịn chỉ tác động lên bước cập nhật gradient sau khi đã tính xong đạo hàm, hai cơ chế này hoạt động hoàn toàn độc lập, loại trừ nguy cơ triệt tiêu lẫn nhau.
> 
> Nhìn vào **Bảng 1 về kết quả xếp hạng**, cải tiến v1-R mang lại một số chuyển biến tích cực:
> - Trên tập Amazon Baby, chỉ số xếp hạng đầu bảng Recall@1 tăng từ 0.0113 lên **0.0125** (tăng tương đối +10.6%).
> - Trên Amazon Sports, Recall@1 đạt 0.0149 (tăng +4.2%), và trên Electronics đạt 0.0097 (tăng +3.2%).
> 
> Tiếp theo, cuộn xuống **Bảng 3 về hồ sơ tài nguyên phần cứng**, cô có thể thấy v1-R kiểm soát bộ nhớ rất tốt:
> - Trên tập Baby, đỉnh VRAM phân bổ giảm từ khoảng 165 MB của baseline xuống còn **136.9 MB**.
> - Trên tập Electronics quy mô 63.000 sản phẩm, VRAM giảm từ 1420 MB xuống còn **1261.2 MB**, tốc độ huấn luyện đạt 39.7 giây mỗi epoch.
> 
> **Tuy nhiên, điểm hạn chế của v1-R là gì?**
> Nếu nhìn kỹ lại Bảng 1, các chỉ số diện rộng như Recall@20 và NDCG@20 trên tập Baby lại giảm nhẹ (Recall@20 giảm từ 0.1042 xuống 0.1027). Qua phân tích động lực học hội tụ, nhóm em nhận thấy việc đối sánh trực tiếp bằng tích vô hướng Euclidean hoặc Cosine giữa các tầng mà chưa xét đến độ trễ pha tự nhiên của các phép tích chập đồ thị đã tạo ra ràng buộc hình học quá cứng, khiến các tầng bị kéo lại gần nhau một cách gượng ép. Đây chính là lý do thúc đẩy nhóm em nghiên cứu phiên bản v2.1 tiếp theo ạ."

---

## PHẦN 3: CẢI TIẾN 2 — STAIR-BCCR v2.1 (ĐIỀU CHUẨN TRONG KHÔNG GIAN PHỨC)
*(Thời lượng: ~2.5 phút)*

### Chỉ dẫn thao tác trên PDF
> Cuộn xuống **Mục 3.2 (trang 8)**, trỏ vào đoạn phân tích nguyên nhân lỗi của bản v2, **Công thức (12) phục hồi FSC**, **Công thức (13)-(16) ánh xạ pha và phép quay Givens**, dừng lại ở **Bảng 4** và **Bảng 5**.

### Lời thoại trình bày (Script nói)
> "Thưa cô, sang **Mục 3.2 là phiên bản STAIR-BCCR v2.1**. Ở giai đoạn này, mục tiêu của nhóm em là thiết lập một hàm mục tiêu phụ trợ nhằm điều chuẩn quan hệ giữa các tầng tích chập mà không làm biến dạng cấu trúc gốc.
> 
> Trước khi ra đời bản v2.1, phiên bản thử nghiệm v2 từng gặp phải một sự cố sụt giảm nghiêm trọng, khi Recall@20 giảm tới 32% đến 41% so với baseline. Khi rà soát chi tiết mã nguồn, nhóm em đã phát hiện ra nguyên nhân cốt lõi: công thức tính toán của mạng FSC trong bản v2 bị nhân nhầm hệ số $a_j$ thay vì $b_j$ ở tử số, khiến trọng số đóng góp của các chiều đặc trưng bị lệch theo bình phương tỷ lệ nghịch $(1-b_j)/b_j$, làm sai lệch hoàn toàn không gian nhúng.
> 
> Trong phiên bản STAIR-BCCR v2.1, nhóm em đã giải quyết triệt để vấn đề này qua ba cơ chế toán học như cô thấy ở các công thức từ (12) đến (16):
> 1. **Phục hồi chuẩn hóa FSC:** Công thức (12) bảo đảm biểu diễn đầu ra khớp hoàn toàn với mô hình STAIR gốc, giúp mô hình phục hồi trọn vẹn mức hiệu năng của baseline.
> 2. **Ánh xạ pha có chặn Bounded Phase Encoder:** Vector tầng 0 và tầng 1 được chuẩn hóa L2 và chuyển sang không gian số phức với độ dài đơn vị $\|\psi(x)\|_2 = 1.0$, loại bỏ hiện tượng bão hòa gradient.
> 3. **Phép quay Givens hai chiều kết hợp Stop-gradient:** Sử dụng toán tử quay Givens $G_\theta$ để học độ lệch pha giữa hai tầng, đồng thời đặt toán tử stop-gradient lên vector tầng 1 để hàm mục tiêu phụ trợ không gây xung đột đạo hàm với hàm BPR chính.
> 4. Ngoài ra, độ tương đồng Hermitian được tính trực tiếp bằng 2 phép nhân ma trận hai chiều GEMM trên phần thực và phần ảo, tránh việc cấp phát tensor 3 chiều kích thước $B \times B \times d$.
> 
> Khi nhìn vào **Bảng 4 về đối chuẩn thực nghiệm**, phiên bản v2.1 đã chứng minh tính đúng đắn khi khắc phục hoàn toàn sự cố sụt giảm:
> - Trên tập Amazon Sports, Recall@20 hồi phục và đạt **0.1115** (vượt baseline 0.1111), NDCG@20 đạt **0.0501** (vượt baseline 0.0500).
> 
> **Tuy nhiên, bài học thực nghiệm lớn nhất ở v2.1 nằm ở Bảng 5 về chi phí tính toán:**
> Mặc dù VRAM tensor phân bổ chỉ ở mức 291 MB, nhưng thời gian huấn luyện mỗi epoch trên Sports tăng từ 5.6 giây lên **18.2 giây** (tăng gấp 3.2 lần). Tổng thời gian chạy thực tế khi bật BCCR tăng từ 6 đến 7.5 lần so với lúc không bật. Phân tích chỉ ra rằng việc tính toán tích vô hướng phức trên từng mini-batch trực tuyến tạo ra chi phí tính toán $O(B^2)$ quá lớn, gây tắc nghẽn thông lượng GPU. Bài học này cho nhóm em thấy rằng: bất kỳ cơ chế can thiệp động nào trong từng mini-batch đều phải đánh đổi bằng chi phí phần cứng rất nặng nề ạ."

---

## PHẦN 4: CẢI TIẾN 3 — STAIR-MHD v3 (MULTI-HEAD HYPERGRAPH DISENTANGLEMENT)
*(Thời lượng: ~2.5 phút)*

### Chỉ dẫn thao tác trên PDF
> Cuộn xuống **Mục 3.3 (trang 13)**, trỏ vào **Công thức (19)-(22) ma trận liên thuộc và mạng gating**, **Công thức (23)-(24) lan truyền nhân tử hóa hai bước**, lướt qua **Hình 3 (TikZ v3)**, rồi dừng lại ở **Bảng 8** và **Bảng 9**, kèm biểu đồ phân kỳ trọng số cổng ở Mục 3.3.4.

### Lời thoại trình bày (Script nói)
> "Để giải quyết hạn chế của việc chỉ xét từng cặp cạnh độc lập, ở **Mục 3.3 nhóm em đề xuất kiến trúc STAIR-MHD v3**, chuyển từ đồ thị cặp sang siêu đồ thị đa đầu có phân tách đặc trưng.
> 
> Ý tưởng xuất phát từ thực tế: mô tả văn bản của sản phẩm thường chứa nhiều từ khóa chung chung dễ tạo ra các liên kết ngữ nghĩa giả, trong khi hình ảnh phản ánh trực tiếp kiểu dáng thực tế. Do đó, như mô tả từ Công thức (19) đến (24) và Hình 3:
> - Nhóm em xây dựng hai ma trận liên thuộc thưa độc lập $H_{\text{text}}$ và $H_{\text{vis}}$ cho hai phương thức. Mỗi sản phẩm tâm và các láng giềng gần nhất tạo thành một siêu cạnh.
> - Thiết kế một mạng nơ-ron Gating hai tầng nhận 4 đặc trưng thống kê tĩnh kết hợp với tín hiệu gắn kết đồng mua $C_e$ và độ tin cậy $\rho_e$. Mạng Gating này tự động học trọng số mềm $w_e$ cho từng siêu cạnh.
> - Để không bị tràn bộ nhớ $O(N^2)$, nhóm em áp dụng kỹ thuật lan truyền nhân tử hóa hai bước qua hai phép nhân ma trận thưa liên tiếp, đưa độ phức tạp về bậc tuyến tính $O(N \cdot k \cdot d)$.
> - Toàn bộ hệ thống được huấn luyện kết hợp giữa BPR, hàm đối chiếu siêu đồ thị HCL và hàm phạt ngân sách gating.
> 
> Nhìn vào **Bảng 8 về kết quả thực nghiệm**, STAIR-MHD v3 đã tạo nên một bước bứt phá rất ấn tượng:
> - Trên tập Amazon Sports có độ thưa lên tới 99.95%, Recall@20 đạt tới **0.1129** (vượt xa baseline 0.1111, tăng tương đối +1.62%); NDCG@20 đạt **0.0508** (vượt baseline 0.0500, tăng +1.60%).
> - Trên tập Baby, mô hình duy trì ổn định với Recall@20 đạt 0.1038 và NDCG@20 đạt 0.0455.
> - Đặc biệt, tại Mục 3.3.4, phân tích động lực học hội tụ chỉ ra hiện tượng phân kỳ trọng số cổng rất thú vị: cổng văn bản tự động giảm từ 0.80 xuống khoảng 0.57-0.62 để lọc nhiễu từ khóa, trong khi cổng hình ảnh duy trì ổn định ở mức 0.79-0.80 để giữ lại đặc trưng thị giác tin cậy.
> 
> **Tuy nhiên, hạn chế lớn nhất của v3 lại hiển thị rất rõ ở Bảng 9 về tài nguyên phần cứng:**
> - Trên Amazon Sports, thời gian huấn luyện 500 epoch tốn tới **2.23 giờ** (18.15 giây/epoch).
> - Trên tập Amazon Electronics với 63.000 sản phẩm, thời gian huấn luyện kéo dài tới **8.00 giờ** (63.84 giây/epoch).
> 
> Nguyên nhân là vì trong mỗi epoch, mô hình phải tính toán thêm nhánh tích chập siêu đồ thị hai bước và hàm mất mát tương phản HCL trực tuyến. Điều này đặt ra câu hỏi then chốt: Liệu ta có thể giữ được mức hiệu năng đỉnh cao 0.1129 của v3 mà đưa thời gian huấn luyện và bộ nhớ VRAM quay trở về mức tối ưu như baseline hay không? Câu hỏi này dẫn dắt trực tiếp đến phiên bản hoàn thiện nhất của đề tài: STAIR4-CSGC v4 ạ."

---

## PHẦN 5: CẢI TIẾN 4 — STAIR4-CSGC v4 (CONFIDENCE-SHRUNK BEHAVIORAL GRAPH CALIBRATION)
*(Thời lượng: ~4.5 phút — Trọng tâm đột phá của báo cáo)*

### Chỉ dẫn thao tác trên PDF
> Cuộn xuống **Mục 3.4 (trang 18)**, chỉ vào tiêu đề cải tiến v4. Đi tuần tự qua:
> 1. Đoạn nguyên lý thiết kế và **8 khối toán học** (Công thức 27 đến 40).
> 2. **Bảng 11** (Hệ thống 7 nhánh đối chứng bóc tách).
> 3. **Hình 4 (Sơ đồ TikZ hai cột đối xứng)**: chỉ tay giải thích Cột trái ngoại tuyến vs Cột phải trực tuyến.
> 4. **Bảng 12** (Kết quả đa thế hệ tại checkpoint tối ưu) và **Bảng 13** (Đối soát ghép cặp trực diện V4-B1 vs V4-C).
> 5. Đoạn phân tích **Quy luật tỷ lệ bằng chứng đồ thị (Evidence Fraction Scaling)**.
> 6. **Bảng 14** (Mức tiêu thụ VRAM đo thật và tốc độ thực thi).

### Lời thoại trình bày (Script nói)
> "Kính thưa cô, đây là cải tiến trọng tâm, có đóng góp mới rõ nét nhất và mang lại giá trị thực tiễn cao nhất của đề tài: **STAIR4-CSGC v4**, viết tắt của Confidence-Shrunk Behavioral Graph Calibration.
> 
> Rút kinh nghiệm sâu sắc từ ba phiên bản trước, nhóm em nhận ra rằng các cơ chế can thiệp động trong lúc huấn luyện trực tuyến luôn kéo theo chi phí tính toán rất đắt đỏ. Nhóm em quyết định quay về nguyên lý tối giản của kỹ thuật (Occam's razor): **Precomputed Static Calibration**. Toàn bộ quá trình hiệu chỉnh đồ thị được thực hiện đúng một lần duy nhất trước khi huấn luyện bằng dữ liệu tập train, sau đó cung cấp toán tử tĩnh $S_\alpha$ cho bộ tối ưu AdamWSEvo. Vòng lặp huấn luyện trực tuyến giữ nguyên 100% cấu trúc của baseline, không thêm bất kỳ tham số học hay hàm mất mát phụ trợ nào.
> 
> Như cô có thể thấy ở các công thức từ (27) đến (40) trong Mục 3.4, kiến trúc v4 được xây dựng dựa trên 4 trụ cột toán học chặt chẽ:
> 
> 1. **Hỗ trợ hành vi điều hòa (Công thức 27-29):** Để tránh việc những người dùng mua sắm quá nhiều tạo ra các liên kết giả giữa các mặt hàng không cùng danh mục, nhóm em gán trọng số nghịch đảo bậc người dùng $w_u = 1 / \max(1, d_u)$ để tính độ tương đồng weighted behavioral cosine $s_{ij}$.
> 
> 2. **Mức hỗ trợ hiệu dụng và hệ số thu nhỏ tin cậy Shrinkage (Công thức 30-31):** Đây là đóng góp khoa học cốt lõi nhất của nhóm em. Trong thương mại điện tử, hai sản phẩm không có tương tác đồng thời trong tập train là do dữ liệu quá thưa thớt, tức là **thiếu bằng chứng quan sát**, chứ không đồng nghĩa chúng là liên kết nhiễu.
>    Công thức thu nhỏ tin cậy $r_{ij}$ bảo đảm rằng: khi không có tương tác đồng thời ($c_{ij} = 0$), $r_{ij}$ suy giảm tuyệt đối về 0. Khi đó, trọng số cơ sở của cạnh đa phương thức ban đầu được giữ nguyên $100\%$. Cơ chế này giúp bảo vệ tối đa các sản phẩm ở dải đuôi dài (long-tail items), hoàn toàn khác biệt với các phương pháp cắt tỉa cạnh thô bạo (hard pruning) trong các công trình gần đây như EVEN hay FREEDOM.
> 
> 3. **Phân tầng bậc và Midrank Empirical CDF (Công thức 32-34):** Phân chia sản phẩm theo 4 phân vị logarit bậc và dùng hàm phân phối tích lũy thực nghiệm thứ hạng giữa để chuẩn hóa điểm số về khoảng đối xứng $[-1, 1]$, xử lý triệt để sự chênh lệch phân bố giữa sản phẩm phổ biến và sản phẩm ít tương tác.
> 
> 4. **Trọng số có cận và toán tử hòa trộn lồi (Công thức 35-37):** Trọng số ma trận kề $W_r$ được khống chế chặt trong khoảng an toàn $[0.5, 1.5]$ lần trọng số ban đầu, không thêm cạnh mới, không xóa cạnh cũ. Ma trận chuẩn hóa đối xứng $S_r$ được hòa trộn lồi với ma trận gốc $S_0$ theo hệ số $\alpha = 0.25$: $S_\alpha = (1 - \alpha)S_0 + \alpha S_r$.
> 
> Tiếp theo, cô nhìn vào **Hình 4 - sơ đồ kiến trúc TikZ**:
> - Cột màu xanh bên trái là toàn bộ quá trình hiệu chỉnh tĩnh ngoại tuyến.
> - Cột màu cam bên phải là quá trình huấn luyện trực tuyến: nhánh forward pass giữ nguyên toàn bộ cấu trúc FSC và hàm mất mát BPR của baseline STAIR. Toán tử $S_\alpha$ chỉ được chuyển sang đúng một vị trí duy nhất: bộ làm mịn Neumann BSC của AdamWSEvo ở backward pass để cập nhật tham số sản phẩm.
> - Đặc biệt, ở Công thức (38)-(39), do $S_0$ và $S_r$ có cùng cấu trúc chỉ số thưa, nhóm em cộng trực tiếp mảng giá trị vào **đúng một ma trận CSR duy nhất** trước khi train. Trong mỗi bước làm mịn, mô hình chỉ thực hiện **đúng một phép nhân ma trận thưa 1-SpMM**: $X^{(l+1)} = S_\alpha X^{(l)}$. Không có tính toán song song, không cấp phát thêm bộ nhớ.
> 
> Để thẩm định độc lập từng thành phần, nhóm em xây dựng hệ thống 7 nhánh đối chứng bóc tách ở **Bảng 11**, bao gồm nhánh kiểm soát Gate-0 V4-B1 ($\alpha=0.0$), nhánh can thiệp hoàn chỉnh V4-C ($\alpha=0.25$), các nhánh bóc tách không shrinkage (V4-NS), không phân tầng bậc (V4-ND), không trọng số người dùng (V4-NA), nhánh xáo trộn ngẫu nhiên (V4-SH) và nhánh hòa trộn ma trận đơn vị (V4-SM).
> 
> Bây giờ, em xin mời cô xem xét các kết quả thực nghiệm then chốt ở **Bảng 12, Bảng 13 và Bảng 14**:
> 
> **Thứ nhất, về độ chính xác gợi ý (Bảng 12 và Bảng 13):**
> - **Trên tập Amazon Sports siêu thưa:** Cả hai nhánh V4-B1 và V4-C đều cải thiện rõ rệt so với baseline. Nhánh can thiệp V4-C đạt Recall@20 là **0.1129** (vượt baseline 0.1111, tăng tương đối +1.62%) và NDCG@20 đạt **0.0505** (vượt baseline 0.0500, tăng +1.00%). Kết quả này tái lập tương đương mức hiệu năng cao nhất của bản v3 (0.1129) mà hoàn toàn không cần đến mạng Gating hay hàm mất mát tương phản phức tạp nào.
> - **Trên tập quy mô công nghiệp Amazon Electronics (63.000 sản phẩm, 1.69 triệu tương tác):** Tại checkpoint tối ưu Epoch 450, nhánh can thiệp V4-C **vượt trội trước nhánh kiểm soát V4-B1 trên toàn bộ 4 chỉ số xếp hạng kiểm định độc lập** (Recall@10 đạt 0.0438, Recall@20 đạt 0.0658, NDCG@10 đạt 0.0244, NDCG@20 đạt 0.0301).
> - **Trên tập Amazon Baby:** Tại checkpoint tối ưu Epoch 325, V4-C giúp chỉ số Top-1 Recall@1 tăng vượt trội đạt 0.0118 (+2.77% so với V4-B1) và NDCG@10 đạt 0.0353 (+0.40%). Đến Epoch 500, mô hình đạt Recall@20 ở mức 0.1038 và NDCG@20 ở mức 0.0454, tiệm cận hoàn toàn mức baseline.
> 
> **Thứ hai, phát hiện khoa học về Quy luật tỷ lệ bằng chứng (Evidence Fraction Scaling):**
> Nhóm em tính toán được tỷ lệ cạnh kNN có tương tác đồng thời trong tập train giảm dần theo quy mô danh mục: từ **10.45%** trên Baby xuống **10.05%** trên Sports và chỉ còn **7.25%** trên Electronics. Điều này có nghĩa là trên tập Electronics, có tới **92.75% số cạnh ngữ nghĩa không có tương tác đồng thời**. Nếu áp dụng cắt tỉa cạnh như các nghiên cứu khác, hơn 92% tri thức đa phương thức sẽ bị xóa bỏ. Cơ chế Shrinkage $r_{ij} \to 0$ giữ nguyên hệ số $1.0\times$ cho nhóm cạnh này chính là chìa khóa khoa học giúp v4 thành công trên tập dữ liệu lớn.
> 
> **Thứ ba, hiệu năng tài nguyên phần cứng vượt trội ở Bảng 14:**
> - **Zero VRAM Overhead tuyệt đối (0.00% chênh lệch):** Mức tiêu thụ VRAM của nhánh can thiệp V4-C hoàn toàn trùng khớp từng byte với nhánh kiểm soát V4-B1 trên cả 3 tập dữ liệu (Baby: 178.27 MiB, Sports: 339.23 MiB, Electronics: 1092.45 MiB). Trên Electronics, mô hình chỉ chiếm vỏn vẹn **7.3% dung lượng của GPU Tesla T4 16GB**.
> - **Tốc độ thực thi siêu tốc:** Nhờ kỹ thuật 1-SpMM, tốc độ huấn luyện của v4 nhanh gấp **3.16 lần** trên Baby, **4.28 lần** trên Sports và **2.24 lần** trên Electronics so với v3. Trên tập Electronics, thời gian huấn luyện 500 epoch giảm từ 8.00 giờ xuống còn **4.60 giờ**, giúp tiết kiệm tới **42.5% chi phí điện toán GPU** mà vẫn bảo toàn trọn vẹn chất lượng gợi ý ạ."

---

## PHẦN 6: TỔNG KẾT VÀ BÀI HỌC THIẾT KẾ HỌC THUẬT
*(Thời lượng: ~2 phút)*

### Chỉ dẫn thao tác trên PDF
> Cuộn đến cuối Mục 3.4, chuẩn bị sẵn sàng cho phần hỏi đáp và nhận xét từ giảng viên.

### Lời thoại trình bày (Script nói)
> "Kính thưa cô, nhìn lại toàn bộ hành trình nghiên cứu Giai đoạn 4 từ Mục 3.1 đến Mục 3.4 trên báo cáo, đề tài đã đi qua một quá trình tiến hóa có tính quy luật rất rõ ràng:
> 
> 1. Từ v1-R và v2.1, nhóm em nhận diện được rào cản chi phí tính toán khi đưa các hàm mục tiêu đối sánh chéo tầng phức tạp vào từng mini-batch trực tuyến.
> 2. Sang v3, mô hình siêu đồ thị đã chứng minh tính ưu việt của việc điều kiện hóa cấu trúc bằng hành vi, xác lập kỷ lục hiệu năng trên tập thưa Sports (0.1129) nhưng phải trả giá bằng thời gian huấn luyện kéo dài.
> 3. Và cuối cùng, phiên bản v4 STAIR4-CSGC đã giải quyết trọn vẹn bài toán cân bằng: tái lập đỉnh cao hiệu năng của v3 trên Sports, chiến thắng toàn diện trên tập công nghiệp Electronics, đồng thời triệt tiêu 100% chi phí bộ nhớ phụ trợ (0.00% VRAM overhead) và tiết kiệm 42.5% thời gian huấn luyện GPU.
> 
> Nghiên cứu chứng minh rằng: một cơ chế hiệu chỉnh đồ thị tĩnh có cơ sở toán học vững chắc kết hợp với nguyên lý thu nhỏ tin cậy hoàn toàn có thể mang lại hiệu năng tương đương hoặc vượt trội so với các mạng nơ-ron phụ trợ phức tạp, đồng thời mở ra khả năng ứng dụng thực tế rất cao trên các hệ thống gợi ý quy mô lớn trong công nghiệp.
> 
> Em xin chân thành cảm ơn cô đã theo dõi phần trình bày báo cáo của nhóm em. Nhóm em rất mong nhận được những câu hỏi, góp ý và định hướng của cô để tiếp tục hoàn thiện đề tài trong giai đoạn bảo vệ khóa luận sắp tới ạ."

---

## BỘ CÂU HỎI PHẢN BIỆN TIỀM NĂNG VÀ HƯỚNG DẪN TRẢ LỜI CỦA SINH VIÊN

### Câu 1: Tại sao trong v4 các em lại can thiệp vào bộ làm mịn AdamWSEvo (BSC) mà không can thiệp trực tiếp vào Forward Stepwise Convolution (FSC)?
> **Gợi ý trả lời:**  
> "Dạ thưa cô, nhánh Forward Stepwise Convolution trong STAIR thực hiện việc lan truyền biểu diễn trên đồ thị tương tác hai phía giữa người dùng và sản phẩm $A$. Nếu ta thay đổi cấu trúc của $A$ trong forward pass, ta sẽ trực tiếp làm biến dạng không gian biểu diễn cộng tác cơ sở và dễ gây ra hiện tượng làm mịn quá mức (oversmoothing) trong quá trình suy diễn.  
> Ngược lại, bộ làm mịn Neumann BSC trong thuật toán AdamWSEvo chỉ tác động lên hướng cập nhật gradient trong backward pass của sản phẩm. Việc hiệu chỉnh đồ thị kNN ngữ nghĩa ở vị trí này đóng vai trò như một bộ lọc không gian, giúp chia sẻ hướng cập nhật giữa các sản phẩm có quan hệ hành vi tương đồng mà không làm thay đổi hàm dự đoán hay giá trị biểu diễn trực tiếp ở forward pass. Điều này vừa giúp gradient hội tụ tốt hơn, vừa bảo toàn trọn vẹn tốc độ suy diễn và cấu trúc của baseline ạ."

### Câu 2: Cơ chế Shrinkage (thu nhỏ tin cậy) trong v4 khác biệt bản chất thế nào so với các phương pháp Graph Pruning (cắt tỉa đồ thị) trong các bài báo gần đây như EVEN hay FREEDOM?
> **Gợi ý trả lời:**  
> "Dạ thưa cô, các phương pháp như EVEN hay FREEDOM thường áp dụng quy tắc cắt tỉa cạnh cứng (hard thresholding hoặc top-k pruning): nếu hai sản phẩm không có tương tác đồng thời từ người dùng thì cạnh ngữ nghĩa giữa chúng sẽ bị loại bỏ hoặc giảm trọng số mạnh.  
> Tuy nhiên, như nhóm em đã phân tích qua thống kê Evidence Fraction Scaling: trên tập dữ liệu quy mô lớn như Amazon Electronics, có tới 92.75% số cạnh kNN ngữ nghĩa không có tương tác đồng thời trong tập train do dữ liệu cực kỳ thưa. Nếu cắt tỉa cạnh, ta sẽ vô tình xóa bỏ hơn 92% thông tin đa phương thức của các sản phẩm đuôi dài (long-tail items).  
> Cơ chế Shrinkage của nhóm em dựa trên nhận thức: thiếu tương tác là do thiếu bằng chứng quan sát, chứ không phải cạnh nhiễu. Khi không có tương tác đồng thời ($c_{ij} = 0$), hệ số tin cậy $r_{ij} = 0$, trọng số ban đầu của cạnh đa phương thức được bảo toàn nguyên vẹn $1.0\times$. Nhờ vậy, mô hình vừa tận dụng được bằng chứng mạnh của nhóm sản phẩm phổ biến, vừa bảo vệ được nhóm sản phẩm ít tương tác ạ."

### Câu 3: Vì sao trên tập Amazon Baby, kết quả Recall@20 của v4 lại thấp hơn baseline một chút (0.1024 so với 0.1042)?
> **Gợi ý trả lời:**  
> "Dạ thưa cô, Amazon Baby là tập dữ liệu có kích thước nhỏ nhất (7.000 sản phẩm) và mật độ tương đối dày (0.117%). Trên tập dữ liệu này:  
> 1. Checkpoint tối ưu của nhánh can thiệp v4-C đạt ở Epoch 325, giúp Recall@1 tăng vượt trội (+2.77% so với nhánh kiểm soát v4-B1) và NDCG@10 tăng +0.40%. Điều này cho thấy mô hình sắp xếp các sản phẩm có độ liên quan cao nhất lên đầu danh sách rất tốt.  
> 2. Đến Epoch 500, khi mô hình hội tụ hoàn toàn, Recall@20 của v4 đạt 0.1038 và NDCG@20 đạt 0.0454, tiệm cận hoàn toàn mức 0.1042 của baseline.  
> 3. Nguyên nhân v4 phát huy hiệu quả cao nhất ở Sports và Electronics là vì cơ chế hiệu chỉnh đồ thị hành vi được thiết kế đặc thù để giải quyết bài toán dữ liệu thưa và danh mục lớn. Ở tập Baby, do mật độ đã tương đối đủ nên biên độ tăng trưởng không rõ rệt như trên tập Sports có độ thưa lên tới 99.95% ạ."
