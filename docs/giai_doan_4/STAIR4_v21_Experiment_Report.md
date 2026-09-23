# STAIR4-v2.1 — Báo cáo thực nghiệm đã đối soát và phản biện phương pháp

Ngày cập nhật: 23/09/2026. Phạm vi: Amazon Baby và Sports, cấu hình C2, seed=1, 500 epochs. Trạng thái: **kết quả quan sát được; chưa xác nhận cải thiện nhiều seed; chưa chạy Electronics v2.1 trong bằng chứng được cung cấp**.

Áp dụng academic-research-suite cho phân tích bằng chứng và phản biện phương pháp. Đây là review nội bộ cùng ngữ cảnh, không phải hội đồng độc lập, không phải kết luận acceptance của một venue; NOT_CALIBRATED. Phiên bản này thay thế những bảng diễn tiến và kết luận thiếu căn cứ trong bản báo cáo trước.

## 1. Kết luận điều hành

1. v2.1 phục hồi phần lớn mức suy giảm của v2, nhưng **chưa chứng minh BCCR cải thiện ranking nhất quán so với baseline**. Baby giảm trên cả bốn metric @10/@20; Sports tăng rất nhỏ ở @20 nhưng giảm ở @10.
2. Chỉ dùng TEST sau khi nạp checkpoint được chọn bằng validation NDCG@20: Baby epoch 325, Sports epoch 460. Không chọn epoch 500 vì nhìn test tốt hơn.
3. Bật BCCR ở epoch 11 đi kèm tăng thời gian mỗi epoch khoảng 7.58 lần trên Baby và 6.19 lần trên Sports so với trung bình epoch 2–10 của cùng run.
4. **Tạm hoãn Electronics full-run với implementation hiện tại.** Có thể benchmark ngắn qua thời điểm bật BCCR nếu cần đánh giá khả năng mở rộng.
5. Nên thực hiện đối chứng B0/B1/C0/C2 có giới hạn trước khi đầu tư lớn vào hướng mới. Đề xuất v4 ở [STAIR4_v4_Report.md](STAIR4_v4_Report.md) là giả thuyết nghiên cứu, chưa có kết quả hoặc bảo đảm vượt baseline.

## 2. Bằng chứng và quy tắc trích xuất

### 2.1. Nguồn

| Nguồn | Vai trò |
|---|---|
| [Baby v2.1](../../logs/GD4/baby_stair4_v21.log) | 500 TRAIN epochs; TEST sau Load best model; config và timing |
| [Sports v2.1](../../logs/GD4/sports_stair4_v21.log) | Tương tự Baby |
| [Baseline Baby](../../logs/paper/baby.log) | Best checkpoint epoch 455 |
| [Baseline Sports](../../logs/paper/sports.log) | Best checkpoint epoch 500 |
| [Learning curve Baby](../../logs/GD4/learning_curve_baby.png) | Kiểm tra hình với log, không thay thế số liệu nguồn |
| [NVML Baby](<../../logs/GD4/vram_profile_baby (3).png>) | Bộ nhớ toàn GPU, không phải tensor allocated |
| [Thiết kế v2.1](STAIR4_v2_1_Report.md) | Định nghĩa baseline contract, ablation và giả thuyết |
| [Head hiện tại](../../models/stair4_heads.py), [utils](../../models/stair4_v2_utils.py) | Phân tích ứng viên bottleneck; không chứng minh exact source của run |

SHA-256 của hai file log đã đọc:

- Baby: c675724f5c45aae8b0ab8bb849dd24f52cfd0da383b249df595d69be171b38c7
- Sports: 6a17abae9095f85e4bbc7362b47058fec3cd6930e601266ee064e13953446802

Các logs mới còn là untracked artifacts tại thời điểm audit. Cần lưu cùng run manifest/source diff trước khi công bố. Chưa tìm thấy diagnostics.jsonl và manifest của hai run C2 trong tập artifact đã đối soát; không suy ra gradient conflict hay exact code revision từ báo cáo cũ.

### 2.2. Quy tắc

- Primary metric: TEST NDCG@20 ở checkpoint tối đa VALID NDCG@20.
- Secondary: Recall@20, Recall@10, NDCG@10; báo đầy đủ, không đổi metric chính sau khi thấy kết quả.
- TRAIN LOSS là total loss trong engine hiện tại. Khi CL bật, không đồng nhất nó với BPR riêng.
- Relative delta = 100 × (new − baseline) / baseline. Delta ở bảng ranking dựa trên số liệu console làm tròn bốn chữ số.
- Bảng summary trước Load best model không phải kết quả final selected checkpoint.
- Trung bình timing tính từ từng dòng Coach.train; epoch 1 loại khỏi warm-up timing vì overhead khởi động.
- Historical baseline không phải một paired rerun hoàn toàn cùng software/hardware. Dùng làm reference; chưa dùng để kết luận nhân quả hoặc độ tin cậy thống kê.

### 2.3. Cấu hình được log ghi nhận

C2 sử dụng hybrid kernel, identity rotation, baseline smoother, eta=0.25, kappa=0.5, temperature=0.2, lambda_max=1e-4, warmup=10, ramp=20, xi=0, query_chunk_size=256, seed=1. Ranking full, retain_seen=False, chọn best theo NDCG@20.

Baby/Sports batch_size=1024. Config Electronics kế thừa batch_size=4096; không được lấy số batches tại 1024 để khẳng định runtime của config 4096.

C2 không huấn luyện Givens và không thử residual spectral. Hai logs này không cung cấp kết luận thực nghiệm về hai module đó.

## 3. Kết quả chính: checkpoint được chọn bằng validation

| Dataset | Metric | STAIR reference | v2.1 C2 | Delta tuyệt đối | Delta tương đối |
|---|---|---:|---:|---:|---:|
| Baby | Recall@1 | 0.0123 | 0.0122 | -0.0001 | -0.81% |
| Baby | Recall@10 | 0.0674 | 0.0661 | -0.0013 | -1.93% |
| Baby | Recall@20 | 0.1042 | 0.1030 | -0.0012 | -1.15% |
| Baby | NDCG@10 | 0.0359 | 0.0355 | -0.0004 | -1.11% |
| Baby | NDCG@20 | 0.0454 | 0.0450 | -0.0004 | -0.88% |
| Sports | Recall@1 | 0.0143 | 0.0144 | +0.0001 | +0.70% |
| Sports | Recall@10 | 0.0743 | 0.0741 | -0.0002 | -0.27% |
| Sports | Recall@20 | 0.1111 | 0.1115 | +0.0004 | +0.36% |
| Sports | NDCG@10 | 0.0405 | 0.0404 | -0.0001 | -0.25% |
| Sports | NDCG@20 | 0.0500 | 0.0501 | +0.0001 | +0.20% |

Baby đạt 98.85% Recall@20 và 99.12% NDCG@20 của reference, không phải 99.6%/99.8% ở selected checkpoint.

Sports best validation NDCG@20 khoảng 0.047825, thấp hơn reference 0.048368 khoảng 1.12%, dù selected TEST nhỉnh hơn. Không dùng riêng test gain để quyết định mở rộng nghiên cứu.

### 3.1. Epoch 500 chỉ dùng làm thông tin phụ

| Dataset | Selected epoch | TEST R@20 selected | TEST N@20 selected | TEST R@20 epoch 500 | TEST N@20 epoch 500 |
|---|---:|---:|---:|---:|---:|
| Baby | 325 | 0.1030 | 0.0450 | 0.1038 | 0.0453 |
| Sports | 460 | 0.1115 | 0.0501 | 0.1112 | 0.0499 |

Baby epoch 500 có test cao hơn epoch 325 không cho phép thay checkpoint hậu nghiệm. Đây là biến thiên giữa validation/test, không chứng minh lỗi chọn checkpoint.

## 4. Diễn tiến hội tụ đúng theo log

### 4.1. Baby

| Epoch | TRAIN total loss | VALID R@20 | VALID N@20 |
|---:|---:|---:|---:|
| 0 | — | 0.0344 | 0.0152 |
| 10 | 0.42441 | 0.0831 | 0.0358 |
| 30 | 0.24089 | 0.0907 | 0.0393 |
| 100 | 0.15884 | 0.0964 | 0.0419 |
| 200 | 0.14201 | 0.0971 | 0.0422 |
| 300 | 0.13713 | 0.0991 | 0.0430 |
| 325 | 0.13624 | 0.0992 | 0.0433 |
| 400 | 0.13476 | 0.0985 | 0.0429 |
| 500 | 0.13387 | 0.0992 | 0.0428 |

Đường cong giảm nhanh đầu run, sau đó có diminishing returns. Sau epoch 325, giảm training loss không tạo best validation mới. Điều này không đủ để quy nguyên nhân cho over-smoothing; cần diagnostics/ablation.

Reference Baby kết thúc với loss 0.13451. Vì vậy diễn giải cũ “baseline ~0.022, v2.1 loss cao hơn 508%” không được log reference này hỗ trợ. Không tính BPR-only delta từ total loss của v2.1.

### 4.2. Sports

| Epoch | TRAIN total loss | VALID R@20 | VALID N@20 |
|---:|---:|---:|---:|
| 0 | — | 0.0511 | 0.0223 |
| 10 | 0.24732 | 0.0885 | 0.0386 |
| 30 | 0.09672 | 0.0958 | 0.0417 |
| 100 | 0.04028 | 0.1016 | 0.0447 |
| 200 | 0.03109 | 0.1057 | 0.0463 |
| 300 | 0.02798 | 0.1060 | 0.0468 |
| 400 | 0.02672 | 0.1072 | 0.0476 |
| 460 | 0.02621 | 0.1077 | 0.0478 |
| 500 | 0.02560 | 0.1070 | 0.0477 |

Sports tiếp tục cải thiện validation muộn. Không dùng pilot 100 epochs để khẳng định bất kỳ kiến trúc nào đã hết khả năng cải thiện. Pilot chỉ phục vụ loại lỗi, đo chi phí và sàng lọc ban đầu.

Reference Sports loss epoch 500 là 0.02564. Hai loss cuối gần nhau không chứng minh embedding hoặc ranking tương đương.

## 5. Chi phí tính toán và bottleneck

### 5.1. Timing đo thật

| Dataset | Mean epoch 2–10 | Mean epoch 11–30 | Tỷ lệ | Mean epoch 31–500 | Coach.fit |
|---|---:|---:|---:|---:|---:|
| Baby | 1.84199 s | 13.96008 s | 7.58× | 13.61115 s | 6767.751071 s = 1.88 h |
| Sports | 4.21895 s | 26.12401 s | 6.19× | 26.05764 s | 12976.236445 s = 3.60 h |

Historical reference fit: Baby 960.027610 s (~16.0 phút), Sports 2210.938315 s (~36.8 phút). Chênh lệch historical timing có confound môi trường; thay đổi ngay epoch 11 trong cùng run là bằng chứng mạnh hơn về overhead khi nhánh CL được kích hoạt.

Lambda nhỏ chỉ scale gradient/loss; không giảm lượng pairwise computation.

### 5.2. Ứng viên bottleneck từ source hiện tại

1. Candidate IDs chuyển CPU; positive masks dùng CPU searchsorted trên train keys.
2. Hai hướng loss trên tập user/item deduplicated; số cặp có thể gần O(B²).
3. Target encoding và rotation lặp lại theo query chunks.
4. Quantile, entropy và nhiều .item()/tensor boolean gây chi phí diagnostics và đồng bộ.
5. Autograd của nhiều chunk được giữ đến backward; chunking giảm kích thước phép GEMM tức thời nhưng không tự bảo đảm tổng saved tensors chỉ O(chunk × B).
6. Gradient conflict diagnostics có thêm autograd.grad ở một số step.

Chưa có profiler để phân bổ phần trăm overhead. Không tuyên bố CPU là bottleneck duy nhất hoặc complex arithmetic là nguyên nhân duy nhất. Log có cảnh báo fork_rng khác cách gọi source hiện tại; càng cần source snapshot của run.

Tối ưu không đổi objective: encode key một lần; lookup candidate trên device phù hợp; log diagnostics theo interval; vector hóa; kiểm tra loss/gradient parity trước và sau. Subsample candidates hoặc tính CL không phải mọi step là **thay đổi estimator/objective**, phải đặt ID riêng.

### 5.3. Bộ nhớ

| Dataset | PyTorch peak allocated | PyTorch peak reserved |
|---|---:|---:|
| Baby | 178.22 (log ghi MB) | 254.00 |
| Sports | 339.92 (log ghi MB) | 488.00 |

Nếu telemetry chia cho 1024² thì đơn vị thực là MiB, cần đồng nhất nhãn ở các run sau. NVML Baby khoảng 891.2 MiB là device memory gồm CUDA context/thư viện và có thể process khác. Không suy ra training nhanh hoặc không có CPU bottleneck từ allocated memory thấp.

Panel schedule trong hình learning curve là đường tái tạo từ config, không phải telemetry per-step. Caption cần phân biệt rõ điều này.

## 6. Phản biện các kết luận nhân quả và thống kê

### 6.1. Đã chứng minh và chưa chứng minh

**Đã quan sát:** v2.1 không còn suy giảm ranking mạnh như run v2 được báo cáo; config C2 có overhead lớn; hai dataset không cho gain nhất quán.

**Chưa chứng minh:** đóng góp riêng BCCR; hiệu quả stop-gradient; phase tốt hơn cosine; significance của gain Sports; xung đột gradient đã biến mất; kết quả Electronics; v3/v4 là SOTA.

Sửa FSC và thay nhiều thành phần đồng thời là một confounded intervention. Để quy hiệu quả cho sửa FSC, cần F1; để quy hiệu quả cho BCCR, cần B1 so C2.

Stop-gradient chặn target branch, nhưng query vẫn dùng embeddings chung với BPR. Chuẩn phase bằng 1 không bảo đảm Jacobian lớn, gradient ổn định hoặc ranking tốt. Fidelity không âm cũng không loại bỏ lực đẩy của softmax CE.

Một seed không ước lượng được seed variance. Không sử dụng ±0.0005 như một khoảng dung sai đo thật. Cần paired seeds và/hoặc per-user ranking scores; bootstrap người dùng không thay thế biến thiên giữa seeds.

### 6.2. Các lỗi được đính chính so với bản cũ

- Thay toàn bộ bảng diễn tiến không khớp log.
- Bỏ baseline Baby loss ~0.022 và “+508%”.
- Sửa Baby Recall@1 reference từ 0.0113 thành 0.0123.
- Không dùng epoch 500 thay selected checkpoint để tăng recovery ratio.
- Bỏ “triệt tiêu hoàn toàn conflict”, “phục hồi 100%”, “chắc chắn không vượt v3”.
- Bỏ dự báo Electronics 21.4 h như một kết quả đo hoặc lower bound.
- Không khẳng định session Kaggle luôn 9h hoặc chắc chắn bị kill; cần kiểm tra runtime/account thực tế.
- Không dùng VALID của một model so với TEST của model khác.
- Không gọi một mô hình nội bộ là SOTA chỉ vì thắng vài phiên bản.
- Không xem run ngắn là “không thể công bố”: có thể báo pilot/anytime với budget công khai, nhưng không thay thế so sánh full-budget.

## 7. Electronics: quyết định có điều kiện

**Hiện tại: NO-GO cho full 500 epochs v2.1 C2. GO có điều kiện cho benchmark kỹ thuật ngắn.**

Sai số dự báo cũ: 1,296,878 / 1024 cho khoảng 1266–1267 batches, trong khi config 4096 chỉ khoảng 316–317. Số thực phụ thuộc drop_last và sampler; phải đọc run. Mặt khác pair count mỗi batch tăng, nên cũng không thể đơn giản chia runtime cũ cho bốn.

Dự báo hợp lệ phải tách:

T_total = T_prepare + N_pre × t_pre + N_post × t_post + N_eval × t_eval + T_checkpoint.

Đo t_pre và t_post qua ít nhất vài epoch hoặc số steps cố định sau startup; benchmark phải thực sự bật BCCR. Báo median/p95 và overhead telemetry. Không suy ra metric từ throughput.

Điều kiện cân nhắc full-run: triển khai đã tối ưu; BCCR có lợi ích validation lặp lại so B1; recipe khóa trước Electronics; đủ budget hoặc resume đã được kiểm chứng. Electronics là kiểm tra external validity, không nên bỏ vĩnh viễn chỉ vì kết quả Baby kém.

## 8. Vòng thử nghiệm tiếp theo với ngân sách giới hạn

Tên ID tuân theo §9.2 báo cáo thiết kế, không theo preset notebook bị đặt lệch ở lần sửa trước:

| ID | Cấu hình | Vai trò |
|---|---|---|
| B0 | main.py baseline | Reference cùng môi trường |
| B1 | Corrected v2.1, CL off, BSC | Gate 0 thực nghiệm |
| F1 | v2 chỉ sửa FSC | Tách nguyên nhân phục hồi; chỉ làm nếu cần luận chứng v2 |
| C0 | Cosine, identity, stopgrad | CL đơn giản |
| C1 | Signed phase eta=0 | Đóng góp phase |
| C2 | Hybrid eta=0.25, identity | Cấu hình đã chạy |
| C3 | Fidelity eta=1 | Đóng góp squared overlap |
| C4 | C2 + learned Givens | Rotation |
| C5 | C2 không target stopgrad | Stopgrad |
| S0/S1 | Residual spectral, CL off/on | Smoother |

Notebook trước đã dùng C0 cho CL off, C1 cho cosine, C3 cho rotation và C4 cho residual: **không khớp thiết kế**. C2 vẫn đúng. Lượt này sửa báo cáo, chưa sửa notebook/code.

Kế hoạch ưu tiên:
1. Gate 0 B0/B1: cùng static tensors, nhiều optimizer steps; sau đó paired rerun để kiểm tra pipeline.
2. Baby B1/C0/C2 pilot 100 epochs, cùng seed và budget, đo throughput/gradient diagnostics.
3. Tối đa hai phương pháp sau sàng lọc, một trong đó là B0, chạy full budget Baby/Sports với paired seeds {1,2,3}; năm seeds nếu nguồn lực cho phép.
4. Dùng validation chọn phương pháp; report mean/std và chênh lệch từng seed. Không loại seed xấu, không dùng test chỉnh recipe.
5. Nếu gain không lặp lại, giữ v2.1 như negative/diagnostic result; chuyển ngân sách sang v4.

## 9. Định vị kết quả trong luận văn

Cách diễn đạt phù hợp:

> Sau hiệu chỉnh implementation và thay đổi regularization, STAIR4-v2.1 đạt hiệu năng gần reference STAIR trên Baby và Sports. Với seed=1, mô hình giảm trên Baby và tăng nhỏ ở một số metrics Sports, đồng thời tăng đáng kể thời gian training khi bật BCCR. Các run hiện có chưa tách được đóng góp của sửa FSC và BCCR; cần đối chứng và nhiều seeds trước khi kết luận ưu thế phương pháp.

Tài liệu này không thay thế kết quả v3, không sửa raw logs, không xác nhận bất kỳ kiến trúc chưa chạy nào sẽ vượt baseline.
