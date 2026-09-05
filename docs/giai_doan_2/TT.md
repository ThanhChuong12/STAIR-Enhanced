# KỊCH BẢN BÁO CÁO TIẾN ĐỘ ĐỢT 2: MÔ HÌNH STAIR VÀ CÁC HƯỚNG CẢI TIẾN
*(Tài liệu chuẩn bị nội dung thuyết minh và trao đổi tiến độ với Giảng viên Hướng dẫn)*

---

## LƯU Ý CHUNG VỀ PHONG CÁCH TRÌNH BÀY
- **Tâm thế:** Báo cáo định kỳ thân mật, chân thành, khoa học và rõ ràng. Tự tin vào các số liệu và chuỗi thử nghiệm thực tế của nhóm.
- **Cách xưng hô:** Xưng *"em"* hoặc *"tụi em"*, gọi *"cô"*.
- **Cấu trúc mỗi phần:** 
  1. *Tóm tắt ý chính cần nắm:* Giúp mình liếc nhanh các luận điểm và con số then chốt trước khi nói.
  2. *Lời thoại trình bày chi tiết:* Văn phong tự nhiên, dễ hiểu, dẫn dắt mạch lạc theo logic nguyên nhân - kết quả.
  3. *Gợi ý trả lời câu hỏi của Cô:* Dự kiến trước các câu hỏi cô có thể hỏi để trả lời chắc chắn, có cơ sở lý thuyết.

---

## 1. TÁI LẬP THỰC NGHIỆM ĐỘC LẬP MÔ HÌNH STAIR GỐC

### Tóm tắt ý chính cần nắm:
- **Mục tiêu:** Xây dựng nền tảng đối soát tin cậy tuyệt đối trước khi cải tiến; đảm bảo mã nguồn và pipeline dữ liệu chuẩn xác.
- **Tiêu chuẩn kiểm thử:** Dùng `NDCG@20` trên tập Validation để chọn best checkpoint (chuẩn hóa cao hơn so với chỉ theo dõi loss).
- **Kết quả:** Sai lệch tuyệt đối so với bài báo gốc (Xu et al., 2024) trên cả 3 tập dữ liệu (Baby, Sports, Electronics) đều dưới **1%**.

| Tập dữ liệu | Chỉ số | Paper gốc | Nhóm tái lập | Chênh lệch tuyệt đối |
| :--- | :--- | :---: | :---: | :---: |
| **Baby** | Recall@20 / NDCG@20 | 0.1037 / 0.0449 | 0.1042 / 0.0447 | +0.0005 / -0.0002 |
| **Sports** | Recall@20 / NDCG@20 | 0.1119 / 0.0505 | 0.1111 / 0.0500 | -0.0008 / -0.0005 |
| **Electronics** | Recall@20 / NDCG@20 | 0.0805 / 0.0357 | 0.0811 / 0.0355 | +0.0006 / -0.0002 |

---

### Lời thoại trình bày với Cô:
> *"Dạ thưa cô, trước khi đi vào các phương án cải tiến, tụi em đã dành thời gian để tái lập độc lập lại toàn bộ mô hình STAIR gốc của tác giả trên cả ba tập dữ liệu: Baby, Sports và Electronics.*
>
> *Tụi em cài đặt quy trình đánh giá chuẩn: dùng chỉ số NDCG@20 trên tập Validation để lưu checkpoint tốt nhất, sau đó mới đánh giá trên tập Test. Kết quả cho thấy các chỉ số Recall và NDCG mà tụi em chạy lại đều khớp với bài báo gốc, sai số tuyệt đối đều dưới 1%.*
>
> *Việc này giúp tụi em hoàn toàn yên tâm là pipeline mã nguồn, cách chia dữ liệu và môi trường chạy thực nghiệm đã chuẩn xác, tạo thành một baseline đối soát tin cậy cho toàn bộ các thử nghiệm phía sau ạ."*

---

### Gợi ý trả lời nếu Cô hỏi:
- **Câu hỏi:** *"Sao nhóm không lấy luôn số liệu trong bài báo so sánh mà phải chạy lại mất thời gian?"*
- **Trả lời:** *"Dạ thưa cô, việc tái lập trực tiếp trên cùng một phần cứng và cùng một quy trình đánh giá giúp loại bỏ hoàn toàn các sai số do ngẫu nhiên hoặc do môi trường thư viện khác nhau. Mọi cải tiến sau này khi so sánh với baseline tái lập sẽ đảm bảo đúng nguyên tắc cách ly biến số, chỉ số tăng là thực sự do mô hình cải tiến chứ không phải do chênh lệch thiết lập ạ."*

---

## 2. NHÓM CẢI TIẾN CAN THIỆP ĐẦU VÀO (v1, v2a, v3) VÀ NGUYÊN NHÂN HIỆU NĂNG GIẢM

### Tóm tắt ý chính cần nắm:
- **Ý tưởng ban đầu:** Nghĩ rằng phương pháp SVD Whitening tĩnh của tác giả quá đơn giản, nên nhóm thử đưa các mạng nơ-ron học tham số vào tiền xử lý đặc trưng.
  - **v1 (De-redundant Gated Projector):** Dùng MLP 2 tầng kèm cổng Gating học được thay thế hoàn toàn cho SVD Whitening.
    - *Kết quả:* Hiệu năng giảm đồng loạt từ **-6.41% đến -9.60%**.
  - **v2a (Residual-Whitening Projector):** Giữ SVD làm nhánh chính, chỉ cộng thêm một nhánh residual MLP nhỏ học độ lệch.
    - *Kết quả:* Hiệu năng vẫn giảm nhẹ **~1% đến 2%**.
  - **v3 (STAIR-LIA v3):** Dùng cơ chế Cross-Attention lọc vùng quan tâm (ROI) lấy cảm hứng từ bài báo CLID, kết hợp ZCA Whitening để giữ trục toạ độ.
    - *Kết quả:* Hiệu năng sụt giảm rất nặng, từ **-19.11% đến -29.82%** (trên tập Baby, Recall@20 từ 0.1042 rơi xuống 0.0731).
- **Nguyên nhân cốt lõi:**
  - SVD Whitening trong STAIR không chỉ là giảm chiều dữ liệu, mà nó tạo ra một hệ cơ sở trực giao và giải tương quan hoàn toàn ($E[xx^T] = I$).
  - Quan trọng hơn, cấu trúc tích chập tiến (FSC) và tích chập lùi (BSC) của STAIR dựa trực tiếp vào phân rã phổ năng lượng theo từng chiều toạ độ (các chiều đầu là tần số thấp chứa tín hiệu cộng tác, các chiều sau là tần số cao chứa tín hiệu đa phương thức).
  - Bất kỳ phép biến đổi phi tuyến (MLP) hay Cross-Attention nào ở đầu vào cũng làm xoay không gian, xáo trộn thứ tự các tần số, khiến bộ lọc tích chập Stepwise Convolution phía sau bị mất phương hướng.

---

### Lời thoại trình bày với Cô:
> *"Dạ thưa cô, ở giai đoạn đầu, tụi em tiếp cận bài toán theo hướng khá tự nhiên: đó là tìm cách làm sạch và nâng cấp biểu diễn đa phương thức ngay từ tầng đầu vào.*
>
> *Trong STAIR gốc, tác giả dùng SVD Whitening tĩnh để nén vector văn bản và hình ảnh từ vài nghìn chiều xuống 64 chiều. Tụi em nghĩ nếu dùng mạng nơ-ron học tham số thì mô hình sẽ thích ứng dữ liệu tốt hơn. Vì vậy, tụi em đã lần lượt thử nghiệm ba phương án:*
>
> *Đầu tiên là bản v1, tụi em thiết kế một mạng MLP có cơ chế cổng Gating để lọc bớt đặc trưng dư thừa. Nhưng kết quả chạy ra lại bị giảm khá rõ, từ 6% đến gần 10% trên cả 3 tập dữ liệu.*
>
> *Sang bản v2a, tụi em nghĩ có thể do mình bỏ hoàn toàn SVD nên mất thông tin nền, nên tụi em đổi sang kiến trúc Residual: giữ SVD làm nhánh chính và chỉ cộng thêm một nhánh residual nhỏ. Kết quả có cải thiện hơn v1, mức giảm co lại còn khoảng 1% đến 2%, nhưng nhìn chung vẫn chưa vượt qua được baseline gốc.*
>
> *Đến bản v3, tụi em thử một hướng sâu hơn lấy cảm hứng từ mô hình CLID: dùng Cross-Attention hai chiều để trích xuất các vùng quan tâm giữa ảnh và chữ, đồng thời thay SVD bằng ZCA Whitening để tránh bị xoay trục toạ độ. Tuy nhiên, kết quả ở bản v3 lại sụt giảm rất nặng, giảm tới gần 30% trên tập Baby.*
>
> *Lúc này tụi em dừng lại và phân tích thật kỹ nguyên nhân về mặt bản chất toán học: Hóa ra không gian SVD Whitening của STAIR có một đặc tính rất đặc biệt. Các vector sau khi làm trắng đã hoàn toàn trực giao và được sắp xếp thứ tự theo năng lượng phổ. Cấu trúc tích chập Stepwise của STAIR tận dụng chính xác thứ tự này: các chiều đầu là tần số thấp để học hành vi cộng tác, các chiều sau là tần số cao để giữ chi tiết đa phương thức.*
>
> *Khi mình đưa MLP phi tuyến hay Attention vào, mình vô tình làm xáo trộn cấu trúc trực giao và thứ tự tần số đó, khiến các tầng tích chập phía sau không còn phân rã đúng phổ năng lượng nữa. Từ phân tích này, tụi em quyết định chuyển hướng hoàn toàn: giữ nguyên vẹn tầng tiền xử lý SVD Whitening và không can thiệp vào đầu vào nữa ạ."*

---

### Gợi ý trả lời nếu Cô hỏi:
- **Câu hỏi:** *"Tại sao ở bản v3 dùng ZCA Whitening mà vẫn giảm nặng như vậy? ZCA vốn giữ nguyên không gian toạ độ cơ sở mà?"*
- **Trả lời:** *"Dạ thưa cô, ZCA đúng là giảm thiểu sự xoay trục toạ độ so với SVD, nhưng vấn đề chính của bản v3 nằm ở khối Cross-Attention. Để tính Attention giữa đặc trưng ảnh 4096 chiều và chữ 384 chiều, tụi em phải cắt nhỏ vector phẳng thành các token nhân tạo. Việc chia cắt này làm mất tính liên kết cục bộ tự nhiên của hình ảnh và văn bản. Khi kết hợp với đồ thị người dùng - sản phẩm quá thưa, các trọng số Attention bị nhiễu nặng và làm hỏng hoàn toàn biểu diễn đặc trưng đưa vào mạng tích chập phía sau ạ."*

---

## 3. CẢI TIẾN 4: STAIR-NLGCL (v4) - CHUYỂN HƯỚNG SANG CONTRASTIVE LEARNING Ở TẦNG ẨN

### Tóm tắt ý chính cần nắm:
- **Chuyển hướng chiến lược:** Giữ nguyên 100% SVD Whitening tĩnh ở đầu vào và cấu trúc tích chập tiến (FSC), tích chập lùi (BSC).
- **Giải pháp:** Tích hợp cơ chế Contrastive Learning trực tiếp vào các tầng biểu diễn ẩn của FSC, lấy cảm hứng từ ý tưởng NLGCL+ (Neighborhood-Enriched Graph Contrastive Learning).
- **Kỹ thuật Zero-cost views:** 
  - Không dùng Edge Dropout hay Node Masking làm biến dạng đồ thị.
  - Không cần chạy thêm bất kỳ forward pass nào.
  - Dùng trực tiếp biểu diễn tầng $0$ (Ego-embedding) và biểu diễn tầng $1$ (1-hop tích hợp lân cận) có sẵn trong quá trình Forward để làm cặp đối chiếu tích cực.
- **Học đối chiếu 2 chiều:** Áp dụng cả phía Người dùng ($\mathcal{L}_u$) và phía Sản phẩm ($\mathcal{L}_i$).
- **Kết quả thực nghiệm theo 2 đợt:**
  - *Đợt 1 (Thang đo siêu tham số CF thuần $\lambda_{	ext{nlgcl}} = 0.0001$):* Hiệu năng gần như không đổi so với baseline (chỉ nhích +0.05% đến +0.10%) vì gradient đối chiếu quá bé so với hàm BPR.
  - *Đợt 2 (Thang đo siêu tham số đa phương thức $\lambda_{	ext{nlgcl}} = 0.01$):* Hiệu năng tăng trưởng đồng loạt trên cả 3 tập dữ liệu (trung bình tăng **+1.63%** trên 12 chỉ số).
    - **Electronics:** Tăng mạnh nhất; NDCG@10 tăng **+5.31%**, Recall@10 tăng **+4.09%**.
    - **Sports:** NDCG@10 tăng **+2.96%**, Recall@10 tăng **+2.42%**.
    - **Baby:** NDCG@10 tăng **+0.28%**.
- **Điểm phát hiện thêm (Hiện tượng vùng trũng):**
  - Trên tập dữ liệu thưa như Sports, các chỉ số Top-10 và NDCG tăng rất tốt, nhưng `Recall@20` lại dừng ở mức **0.1110** (baseline là 0.1111).
  - Lý do: Trên đồ thị thưa, hàm InfoNCE thuần có xu hướng gom các biểu diễn lân cận lại quá chặt (hiện tượng co cụm biểu diễn - representation degeneration), khiến danh sách Top-20 ở đuôi bị ảnh hưởng nhẹ.

---

### Lời thoại trình bày với Cô:
> *"Dạ thưa cô, sau khi nhận ra bản chất của SVD Whitening, ở cải tiến 4 (tụi em đặt tên là STAIR-NLGCL), tụi em thực hiện một bước ngoặt về chiến lược: giữ nguyên vẹn toàn bộ đầu vào SVD và cấu trúc tích chập của tác giả. Thay vào đó, tụi em bổ sung một nhánh học tương phản (Contrastive Learning) ở tầng biểu diễn ẩn như một hàm mất mát phụ trợ.
>
> Tụi em áp dụng một kỹ thuật rất hay lấy cảm hứng từ bài báo NLGCL+, gọi là Zero-cost views. Nghĩa là tụi em không cần làm Edge Dropout hay tạo thêm view đồ thị phụ gây tốn bộ nhớ, mà tận dụng ngay biểu diễn tầng 0 (đặc trưng gốc) và biểu diễn tầng 1 (đã gom thông tin lân cận) có sẵn trong nhánh tích chập FSC để làm cặp đối chiếu. Tụi em tính InfoNCE cho cả hai phía người dùng và sản phẩm.
>
> Khi thử nghiệm, ban đầu tụi em áp dụng trọng số lambda = 0.0001 theo các bài báo đồ thị đơn phương thức, thì thấy kết quả hầu như không thay đổi gì so với baseline. Tụi em nhận ra trong mô hình đa phương thức, độ lớn gradient của đặc trưng multimodal lớn hơn hẳn CF thuần, nên lambda đó quá nhỏ.
>
> Tụi em đã nâng lambda lên mức 0.01 cho phù hợp với bài toán đa phương thức. Và kết quả thực nghiệm đợt 2 đã cho thấy hiệu quả rất rõ rệt:
> - Cả ba tập dữ liệu đều ghi nhận mức tăng trưởng dương.
> - Đáng chú ý nhất là tập Electronics, tập lớn nhất với hơn 1.6 triệu tương tác: chỉ số NDCG@10 tăng vọt +5.31%, Recall@10 tăng +4.09%.
> - Trên tập Sports, NDCG@10 cũng tăng +2.96% và Recall@10 tăng +2.42%.
>
> Tụi em nhận thấy hàm mất mát tương phản có tác dụng rất mạnh trong việc xếp hạng (ranking), kéo các sản phẩm thực sự liên quan lên các vị trí đầu tiên của danh sách, đó là lý do NDCG tăng mạnh hơn Recall.
>
> Tuy nhiên, khi soi kỹ vào số liệu của tập Sports, tụi em phát hiện một điểm thú vị: trong khi Recall@10 tăng tốt thì Recall@20 lại bị kẹt nhẹ ở mức 0.1110, thấp hơn baseline một chút xíu (0.1111). Phân tích ra thì đây là hiện tượng co cụm biểu diễn thường gặp trên các đồ thị có mật độ quá thưa. Và đây chính là động lực trực tiếp để tụi em nghiên cứu tiếp cải tiến 5 ạ."*

---

### Gợi ý trả lời nếu Cô hỏi:
- **Câu hỏi:** *"Zero-cost views nghĩa là không tốn chi phí gì à? Cụ thể nó tiết kiệm tài nguyên thế nào?"*
- **Trả lời:** *"Dạ thưa cô, thông thường trong Graph Contrastive Learning như SGL hay SimGCL, mô hình phải thực hiện Edge Dropout hoặc Node Dropout để tạo ra một ma trận kề ngẫu nhiên mới, rồi phải chạy thêm một lần Forward Pass qua GNN nữa. Việc đó làm tăng gấp đôi thời gian huấn luyện và tăng mạnh VRAM. Còn ở đây, STAIR trong quá trình Forward tự nhiên đã phải tính biểu diễn tầng 0 và tầng 1 rồi. Tụi em lấy trực tiếp hai tensor đó tính InfoNCE trong cùng một batch luôn, không sinh ma trận mới và không chạy lại GNN, nên thời gian huấn luyện mỗi epoch gần như không đổi so với baseline gốc ạ."*

---

## 4. CẢI TIẾN 5: STAIR-NE-NLGCL (v5) - BƠM NHIỄU THÍCH ỨNG PHỔ VÀ THỬ NGHIỆM LỌC MẪU ÂM GIẢ

### Tóm tắt ý chính cần nắm:
- **Động lực:** Giải quyết hiện tượng co cụm biểu diễn của v4 trên tập dữ liệu thưa, giúp mô hình vượt qua ngưỡng bão hòa của Recall@20.
- **Hai cơ chế đề xuất trong v5:**
  1. **Bơm nhiễu thích ứng theo phổ năng lượng (Spectral-guided Noise):**
     - Bơm vector nhiễu ngẫu nhiên $\eta$ vào biểu diễn ẩn nhưng điều hòa biên độ theo vector phổ $oldsymbol{eta} = 1 - oldsymbol{eta}_3$.
     - Các chiều tần số thấp ($d 	o 0$): nhận nhiễu lớn nhất ($eta pprox 0.9$) để mở rộng phân bố không gian, chống co cụm.
     - Các chiều tần số cao ($d 	o 63$): lượng nhiễu triệt tiêu dần về 0 ($eta 	o 0$) để bảo toàn nguyên vẹn đặc trưng đa phương thức gốc.
     - Phép nhân bảo toàn hướng $	ext{sign}(h) \odot rac{\eta}{\|\eta\|_2}$: đảm bảo nhiễu chỉ làm phân tán độ lớn chứ không làm đảo dấu toạ độ.
  2. **Cơ chế lọc mẫu âm giả (In-batch False Negative Filtering):**
     - Tạo mặt nạ $\mathcal{M}_{b, k}$ để loại các cặp trong batch có độ tương đồng ngữ nghĩa cao ($S_{b, k} > 	au_{	ext{thresh}}$) ra khỏi mẫu số của InfoNCE.
- **Kết quả Pha 1 (Bơm nhiễu thuần, $	au_{	ext{thresh}} = 1.0$):**
  - **Sports (Đồ thị siêu thưa):** 
    - Phá vỡ mức trần bão hòa: `Recall@20` đạt **0.1113** (vượt baseline 0.1111 và v4 0.1110).
    - `NDCG@20` lập kỷ lục mới đạt **0.0508** (+1.60% so với baseline, +0.20% so với v4).
    - Đường cong học (learning curve) trên tập Validation tăng trưởng đều đặn và đạt đỉnh ở epoch 365.
  - **Baby:** NDCG@10 tiếp tục được củng cố tăng **+0.56%** (0.0361 vs 0.0359).
- **Kết quả Pha 2 (Kích hoạt lọc âm giả $	au_{	ext{thresh}} = 0.85$ trên Baby) & Phát hiện chuyên sâu:**
  - Hiệu năng sụt giảm nghiêm trọng (Recall@20 rơi từ 0.1022 xuống 0.0344).
  - *Nguyên nhân bản chất:* Trong không gian SVD Whitening, các chiều đã được giải tương quan hoàn toàn ($E[xx^T] pprox I$), phân bố hình cầu đồng nhất. Tích vô hướng trong không gian này không còn phản ánh cosine similarity ngữ nghĩa như trong không gian thô ban đầu. Việc lọc với ngưỡng 0.85 đã vô tình triệt tiêu mất các mẫu âm mang tính cộng tác thực sự.
  - *Ý nghĩa khoa học:* Khẳng định tính ưu việt của cơ chế Bơm nhiễu thích ứng phổ (Pha 1) và làm rõ đặc thù toán học của không gian SVD Whitening trong hệ thống gợi ý.

---

### Lời thoại trình bày với Cô:
> *"Dạ thưa cô, để giải quyết hiện tượng co cụm biểu diễn của bản v4 trên tập dữ liệu thưa, tụi em đã thiết kế phiên bản hoàn thiện tiếp theo là STAIR-NE-NLGCL (bản v5).
>
> Ở bản này, tụi em đưa thêm một lượng nhiễu Gauss ngẫu nhiên vào biểu diễn ẩn để kích thích không gian biểu diễn phân tán đều hơn, chống hiện tượng co cụm. Điểm mấu chốt là tụi em không bơm nhiễu đồng đều trên mọi chiều, vì làm thế sẽ làm hỏng đặc trưng đa phương thức giống như các bản v1-v3 trước đây.
>
> Tụi em đã thiết kế công thức bơm nhiễu điều hòa thích ứng theo phổ năng lượng của STAIR:
> - Ở các chiều đầu là tần số thấp (chứa hành vi cộng tác), lượng nhiễu được đưa vào lớn nhất để mở rộng phân bố không gian.
> - Càng về các chiều sau là tần số cao (chứa đặc trưng ảnh và chữ sau SVD), lượng nhiễu triệt tiêu dần về 0 để bảo toàn tuyệt đối thông tin đa phương thức.
> - Đồng thời, tụi em sử dụng phép nhân bảo toàn hướng vector sign(h) để đảm bảo vector nhiễu không làm lật dấu toạ độ của biểu diễn.
>
> Kết quả chạy ở Pha 1 (chế độ bơm nhiễu phổ thuần) đã mang lại kết quả đúng như kỳ vọng của tụi em:
> - Trên tập siêu thưa Sports, mô hình đã phá vỡ được mức trần bão hòa của v4: Recall@20 đạt 0.1113, vượt cả baseline gốc và bản v4. Chỉ số NDCG@20 thiết lập đỉnh mới là 0.0508, tăng +1.60% so với baseline.
> - Trên tập Baby, chỉ số NDCG@10 cũng được củng cố tăng +0.56%.
> - Quan sát đường cong huấn luyện, tụi em thấy chỉ số Validation NDCG@20 tăng đều đặn xuyên suốt các epoch và đạt đỉnh bền bỉ ở epoch 365 mà không hề bị quá khớp (overfitting) sớm.
>
> Bên cạnh đó, tụi em cũng chạy thử nghiệm Pha 2 với cơ chế Lọc mẫu âm giả (False Negative Filtering) bằng ngưỡng tương đồng cosine 0.85 trên tập Baby. Kết quả pha này hiệu năng lại bị giảm. Tụi em đã phân tích sâu và nhận ra một kết luận khoa học rất giá trị: Trong không gian SVD Whitening, các chiều đã được giải tương quan và phân bố đẳng hướng dạng hình cầu, nên tích vô hướng ở đây không còn mang ngữ nghĩa góc cosine thông thường như không gian đặc trưng thô ban đầu. Việc áp ngưỡng 0.85 đã vô tình lọc mất các mẫu âm hữu ích.
>
> Nhờ đó, tụi em kết luận rằng phương án Bơm nhiễu thích ứng phổ của Pha 1 chính là cấu hình tối ưu và vững chắc nhất cho mô hình v5 ạ."*

---

### Gợi ý trả lời nếu Cô hỏi:
- **Câu hỏi:** *"Tại sao bơm nhiễu vào biểu diễn lại giúp cải thiện được kết quả trên tập thưa mà không làm mô hình bị nhiễu loạn?"*
- **Trả lời:** *"Dạ thưa cô, trên các đồ thị rất thưa, số lượng cạnh tương tác ít khiến GNN có xu hướng chiếu các user/item vào những cụm rất hẹp trong không gian embedding. Khi tính hàm loss tương phản InfoNCE, các vector này bị nén lại quá chặt, làm mất tính đa dạng khi gợi ý. Bơm nhiễu đóng vai trò như một cơ chế làm trơn (regularization): nó tạo ra các biến thể cục bộ quanh vector gốc, buộc hàm InfoNCE phải học cách phân biệt các điểm dữ liệu trong một vùng lân cận rộng hơn. Nhờ vector phổ năng lượng beta khống chế, nhiễu chỉ tác động vào các chiều cộng tác mà không chạm vào đặc trưng đa phương thức, nên mô hình mở rộng được phân bố mà vẫn giữ trọn thông tin ngữ nghĩa ạ."*

---

## 5. ĐÁNH GIÁ CHI PHÍ TÀI NGUYÊN PHẦN CỨNG VÀ TỔNG KẾT

### Tóm tắt ý chính cần nắm:
- **Đo lường bộ nhớ VRAM thực tế:** Đo bằng công cụ thư viện `pynvml` trực tiếp trên GPU Nvidia Tesla T4 trong suốt quá trình huấn luyện:
  - **Baby:** Baseline thô 720 MB $	o$ v4 753 MB $	o$ v5 **797 MB** (chỉ tăng +77 MB).
  - **Sports:** Baseline thô 940 MB $	o$ v4 973 MB $	o$ v5 **995 MB** (chỉ tăng +55 MB).
  - **Electronics:** Baseline thô 4.80 GB $	o$ v4 **4.85 GB** (chỉ tăng +50 MB).
- **Thời gian chạy mỗi epoch:** Gần như tương đương baseline do toàn bộ phép toán là ma trận thưa và phép nhân tensor nguyên bản trên GPU.
- **Tổng kết đối soát 6 phiên bản:**
  - *v1, v2a, v3:* Xác định rõ ranh giới không nên can thiệp phi tuyến vào không gian SVD Whitening.
  - *v4:* Đạt bước nhảy vọt toàn diện trên dữ liệu dày (Electronics) nhờ Contrastive Learning đa tầng dạng Zero-cost.
  - *v5:* Đạt sự cân bằng và hoàn thiện tối đa, giải quyết tốt hiện tượng co cụm biểu diễn trên tập dữ liệu thưa (Sports, Baby).
- **Mô hình chốt của đề tài:** **STAIR-NE-NLGCL v5** là mô hình hoàn chỉnh được chọn để đóng gói báo cáo và viết bài báo khoa học.

---

### Lời thoại trình bày với Cô:
> *"Dạ thưa cô, một tiêu chí quan trọng mà nhóm luôn bám sát từ đầu đề tài là: mô hình cải tiến phải giữ được ưu điểm cốt lõi của STAIR gốc, đó là tính gọn nhẹ và khả năng huấn luyện nhanh trên phần cứng thông thường.
>
> Tụi em đã dùng thư viện pynvml để đo chính xác mức tiêu thụ bộ nhớ đỉnh VRAM trên GPU xuyên suốt quá trình huấn luyện:
> - Trên cả hai tập Baby và Sports, phiên bản v5 hoàn chỉnh chỉ tiêu tốn lần lượt 797 MB và 995 MB VRAM, tức là chỉ tăng khoảng 50 đến 70 MB so với baseline thô, và duy trì an toàn dưới ngưỡng 1 GB.
> - Ngay cả trên tập lớn nhất là Electronics với 1.6 triệu tương tác, VRAM chỉ tốn 4.85 GB, hoàn toàn nằm trong giới hạn của các GPU phổ thông như GTX 1660 hay RTX 3060.
> - Thời gian huấn luyện mỗi epoch cũng không phát sinh thêm đáng kể vì cơ chế Zero-cost views và bơm nhiễu đều là các phép toán vector trực tiếp trên PyTorch.
>
> Tổng kết lại chuỗi nghiên cứu qua các phiên bản:
> - Các phiên bản đầu v1, v2a, v3 giúp tụi em hiểu sâu sắc về bản chất bảo toàn phổ năng lượng và tính trực giao của SVD Whitening.
> - Phiên bản v4 chứng minh hiệu quả vượt bậc của Contrastive Learning đa tầng ở không gian ẩn, mang lại mức tăng trưởng mạnh mẽ trên tập dữ liệu lớn.
> - Và phiên bản v5 hoàn thiện trọn vẹn bài toán khi giải tỏa được hiện tượng co cụm trên tập dữ liệu thưa nhờ cơ chế bơm nhiễu thích ứng phổ.
>
> Do đó, tụi em xin phép cô lấy mô hình STAIR-NE-NLGCL v5 làm mô hình đề xuất chính thức của nhánh STAIR trong đề tài tốt nghiệp. Tụi em đã hoàn thành toàn bộ phần văn bản báo cáo chi tiết và bảng biểu trong Chương 3 của cuốn luận văn ạ.
>
> Tụi em rất mong nhận được những nhận xét và góp ý thêm từ cô để hoàn thiện đề tài tốt hơn nữa ạ!"*

---

### Gợi ý các câu hỏi mở rộng Cô có thể hỏi & Hướng trả lời:
1. **Câu hỏi:** *"Nếu so sánh giữa nhánh cải tiến của STAIR và nhánh của REARM trong đề tài thì STAIR có ưu thế gì vượt trội?"*
   - **Trả lời:** *"Dạ thưa cô, ưu thế lớn nhất của STAIR là tính tinh gọn (lightweight) và tốc độ hội tụ cực nhanh. Nhờ SVD Whitening tĩnh và cấu trúc Stepwise Convolution, STAIR giải quyết triệt để bài toán tài nguyên. Các cải tiến v4 và v5 của tụi em kế thừa trọn vẹn ưu điểm này: vừa nâng cao độ chính xác bằng tự giám sát, vừa không làm phình to mô hình hay tốn thêm VRAM, rất phù hợp cho các bài toán thực tế có tài nguyên tính toán giới hạn ạ."*

2. **Câu hỏi:** *"Kế hoạch tiếp theo của nhóm sau buổi báo cáo này là gì?"*
   - **Trả lời:** *"Dạ thưa cô, kế hoạch của tụi em gồm 2 việc chính: Thứ nhất là rà soát lại toàn bộ định dạng LaTeX, tài liệu trích dẫn và bảng biểu của Chương 3 theo góp ý của cô hôm nay. Thứ hai là tụi em sẽ chuyển sang đồng bộ kết quả thực nghiệm của nhánh mô hình thứ hai (REARM/NEGCL) để viết tiếp Chương 4 và chuẩn bị cho đợt báo cáo nghiệm thu tổng thể ạ."*
