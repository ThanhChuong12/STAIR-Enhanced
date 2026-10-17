# STAIR — Đối soát giai đoạn 4–6 và lộ trình nghiên cứu cho paper

> **Cập nhật16/10/2026:** Bản này lưu kết luận tại14/10, trước khi có kết quả V8.1 và Electronics GĐ6-v2. Quyết định nghiên cứu mới nhất nằm tại [STAIR_ICAART_Research_Audit_20261016.md](STAIR_ICAART_Research_Audit_20261016.md): chọn V8.1-64D làm ứng viên paper có điều kiện, ưu tiên paired multi-seed/ablation và capacity factorial. Không dùng nhận xét “V8.1 chưa chạy” hoặc “GĐ6-v2 thiếu Electronics” của snapshot này như trạng thái hiện tại.

**Ngày đối soát:** 14/10/2026, Asia/Saigon.  
**Kết luận chính:** chọn **STAIR5-v8 / WMSG-CSE, giai đoạn 5** làm nền paper và đối chứng tiếp theo; chưa có phiên bản nào được chứng minh vượt trội thống kê trên cả ba dataset. Giai đoạn6-v1/v2 là các mở rộng đã chạy, với phần tăng thêm rất nhỏ và chưa đầy đủ ablation/multi-seed.

## 1. Phạm vi, phương pháp và độ tin cậy

Áp dụng `academic-research-suite`, targeted `deep-research` và `academic-paper-reviewer` ở methodology-focus. Đây là audit kết quả và nghiên cứu có mục tiêu, không phải systematic review toàn bộ lĩnh vực hay đánh giá một manuscript hoàn chỉnh.

- Index38 file kiến trúc/experiment trong giai đoạn4–6; đọc các phần kết quả, protocol, claim và cơ chế liên quan. Không tuyên bố đọc toàn bộ từng đoạn văn của38 file.
- Một nhiệm vụ source verification riêng đối soát giai đoạn4; một nhiệm vụ methodology riêng đối soát giai đoạn5–6. Các nhiệm vụ không nhận kết quả của nhau trước khi hoàn thành. Tổng hợp và venue fit do lượt chính thực hiện.
- Cùng họ mô hình, **NOT_CALIBRATED**; không phải hội đồng peer review độc lập của con người và không dự báo xác suất được nhận.
- Báo cáo đầu vào là dữ liệu được phản biện. Các từ “SOTA”, “hoàn hảo”, “100%”, “tỷ lệ vàng” không tự tạo thành bằng chứng.
- Không sửa các báo cáo lịch sử, source hoặc chạy training trong audit này. Tạo báo cáo riêng để tránh âm thầm thay đổi lịch sử nghiên cứu.

Receipt: [material_inventory.json](D:/4thY_HCMUS/KLTN/STAIR-Enhanced/reports/trajectory_audit/material_inventory.json). Tám dòng TEST mới được trích trực tiếp từ log sau `Load best model`: [verified_recent_selected_test.csv](D:/4thY_HCMUS/KLTN/STAIR-Enhanced/reports/trajectory_audit/verified_recent_selected_test.csv).

**Thứ tự ưu tiên:** TEST của checkpoint được chọn bằng validation trong raw log/JSONL → manifest cùng run → bảng selected-checkpoint → experiment report → bảng comparator chép lại → dự báo trong architecture report. Không dùng TEST epoch cuối khi best-validation epoch khác, không đổi VALID thành TEST, không ghép số tốt nhất của nhiều run thành một model.

Nhãn dùng dưới đây: **FACT** = kiểm chứng source/artifact; **OBSERVATION** = số đo với phạm vi; **INTERPRETATION** = cách giải thích; **HYPOTHESIS** = cần kiểm định. Các hiệu ứng tính từ bảng làm tròn chỉ là phép tính mô tả.

## 2. Ma trận thực nghiệm đã đối soát

Mỗi ô là **Recall@20 / NDCG@20 của TEST tại checkpoint chọn bằng validation**. Dấu “—” nghĩa là chưa có TEST phù hợp trong corpus, không phải0. Không trộn phiên bản cùng số nhưng khác giai đoạn.

| Giai đoạn / phiên bản | Baby | Sports | Electronics | Trạng thái bằng chứng |
|---|---:|---:|---:|---|
| STAIR tái lập lịch sử | .1042 / .0454 | .1111 / .0500 | .0665 / .0303 | Raw selected logs; không phải control đồng thời của mọi phiên bản |
| GĐ4-v1-R, CNLGCL | .1027 / .0451 | .1124 / .0509 | .0671 / .0309 | Đối soát raw logs; cao nhất N20 trong các GĐ4 hoàn tất |
| GĐ4-v2, A6 | .0606 / .0255 | .0749 / .0331 | — | Suy giảm lớn; có lỗi FSC nên không quy hết cho giả thuyết lượng tử |
| GĐ4-v2.1, C2 | .1030 / .0450 | .1115 / .0501 | — | Phục hồi sau sửa lỗi/thay đổi loss; chưa phân lập đóng góp |
| GĐ4-v3, MHD | .1033 / .0448 | .1129 / .0508 | — | Electronics dừng ở451; số .0670/.0299 là VALID, không phải TEST |
| GĐ4-v4, CSGC-C | .102370 / .044713 | .112874 / .050500 | .065756 / .030063 | Có control B1 cùng run; hiệu ứng N20 rất nhỏ và hỗn hợp |
| GĐ4-v5, RAM | — | — | — | Chỉ có VALID; head chọn epoch0, không có learned-head gain được xác nhận |
| GĐ5-v1, LHC | .1030 / .0447 | .1133 / .0506 | .0666 / .0301 | Theo selected tables gốc; hyperbolic extension không thắng nhất quán |
| GĐ5-v2, C-HET/ET | .1029 / .0447 | .1129 / .0505 | — | ET/control arm; không chứng minh mọi module đề xuất |
| GĐ5-v3, DP-ref | .1029 / .0447 | .1129 / .0505 | — | Reference arm; không chứng minh mọi cơ chế dynamic compatibility |
| GĐ5-v4, NLGCL-CSE | .1056 / .0461 | .1143 / .0517 | .0678 / .0316 | Bước tiến chính; comparator bắt buộc của V8 |
| GĐ5-v5, CAM | — | — | — | Kiến trúc đề xuất; không có experiment report tương ứng |
| GĐ5-v5.1, KPE | .1054 / .0461 | .1136 / .0516 | — | Không cải thiện rõ hơn V4 |
| GĐ5-v5.2, BPE | .1040 / .0457 | .1135 / .0515 | .0678 / .0316 | Giảm Baby/Sports; Electronics gần V4 |
| GĐ5-v6, BCSR | .1056 / .0461 | .1143 / .0517 | .0681 / .0317 | Chủ yếu ngang V4; Electronics có tăng nhỏ theo report |
| GĐ5-v7, UCR-D | .10558 / .0461439 | .11426 / .0516274 | .06773 / .031643 | Dùng ledger selected đúng, không dùng .0520/.0318 chép về sau |
| **GĐ5-v8, WMSG-CSE** | **.1054 / .0462** | **.1142 / .0518** | **.0680 / .0317** | Ba raw selected TEST; static graph, T4 |
| GĐ6-v1, LCAE-WMSG-CSE | .1050 / .0462 | .1142 / .0519 | .0680 / .0317 | Manifest mới xác nhận latent categories đang active trên cả ba |
| GĐ6-v2, AKP-WMSG-CSE | .1054 / .0463 | .1141 / .0518 | — | CAN active; hoàn tất Baby/Sports, chưa Electronics |
| GĐ5-v8-256D / v8.1 / v9 | — | — | — | Có đặc tả/code tùy phiên bản; chưa thấy production experiment tương ứng trong corpus |

GĐ5-v1–v6 dùng selected tables của từng report và kiểm tra chéo trong phạm vi methodology audit; không gán cùng mức xác minh từng dòng như8 raw recent logs. Các giá trị B1/control không được ghi thành thành tích CSGC. V8/V6-v1/V6-v2 log gần đây chỉ in4 chữ số; không bịa thêm precision.

### Các anchor quan trọng

- STAIR lịch sử: `logs/paper/baby.log:2222`, `sports.log:2284`, `electronics.log:2385`.
- GĐ4-v1-R: `logs/GD4/baby_v1_r.log:2322`, `sports_v1_r.log:2346`, `electronics_v1_r.log:2347`.
- GĐ4-v3: `logs/GD4/mhd_v3/baby_mhd_v3.log:1862`; Electronics file kết thúc TRAIN451, VALID ở1692.
- GĐ4-v4: `selected_checkpoint_metrics.csv` của Baby/Sports/Electronics trong `logs/GD4/V4/`; row2=B1, row3=C.
- GĐ4-v5: `logs/GD4/V5/v5_log/reports/all_runs.csv:2–4` có TEST trống; các `evaluation.jsonl` chỉ có mode valid.
- GĐ5-v7: [stair5_v7_manifest.json](D:/4thY_HCMUS/KLTN/STAIR-Enhanced/logs/GD5/stair5_v7_complete_artifacts/stair5_v7_manifest.json:6); [V8 architecture reconciliation](D:/4thY_HCMUS/KLTN/STAIR-Enhanced/docs/giai_doan_5/STAIR5_v8_Report.md:62).
- V8 TEST: Baby log1868, Sports1872, Electronics1882; đường dẫn chính xác nằm trong CSV receipt.
- GĐ6-v1 TEST: Baby1886, Sports1901, Electronics1902; GĐ6-v2 TEST: Baby1883, Sports1891.

## 3. Phiên bản nào đang tốt nhất?

### 3.1 Không có một winner cho mọi tiêu chí

- **N20 Baby theo số đã in:** GĐ6-v2=.0463, cao nhất trong các selected TEST đã đối soát; hơn V8 .0001, khoảng **0.216% tương đối**. Chưa chứng minh ý nghĩa thống kê.
- **N20 Sports:** GĐ6-v1=.0519, hơn V8 .0001, khoảng **0.193% tương đối**. V7 không có .0520 trong ledger selected phù hợp.
- **N20 Electronics:** V8, GĐ6-v1 và GĐ5-v6 cùng khoảng .0317 ở precision hiện có. Không thể xếp thứ tự chắc chắn từ số làm tròn. GĐ6-v2 chưa chạy dataset này.
- **R20:** V4/V6 giữ lợi thế nhỏ Baby/Sports so với V8; GĐ5-v6 báo .0681 Electronics, nhỉnh hơn .0680 của V8. Không lấy N20 winner làm winner mọi metric.
- **Nền paper ưu tiên:** V8, vì ba dataset đã hoàn tất, cơ chế static tương đối nhỏ, không cần metadata ngoài và không có nhánh graph động trong mỗi bước.

**INTERPRETATION:** V8 là lựa chọn cân bằng cho phát triển paper, không phải model được chứng minh thắng thống kê tất cả phiên bản. GĐ6-v1 thêm complexity nhưng chưa tăng R20/N20 nhất quán; GĐ6-v2 có lợi thế Baby một lần chạy và thiếu Electronics/ablation. Chưa có cơ sở thay nền V8 chỉ vì version mới hơn.

### 3.2 Phần tăng thật của V8 đến đâu?

| Dataset | V8 R20 vs STAIR lịch sử | V8 N20 vs STAIR lịch sử | V8 N20 vs GĐ5-v4 | V8 R20 vs GĐ5-v4 |
|---|---:|---:|---:|---:|
| Baby | +1.152% | +1.762% | +0.217% | −0.189% |
| Sports | +2.790% | +3.600% | +0.193% | −0.087% |
| Electronics | +2.256% | +4.620% | +0.316% | +0.295% |

Tính bằng100(new/old−1) từ giá trị đã in. Trung bình ba relative N20 gains vs STAIR là khoảng3.33%, **không phải** một effect size gộp có trọng số hay ý nghĩa thống kê. V8 hơn V4 rất ít: phần tiến bộ chủ yếu đã có từ NLGCL/CSE. Trong paper phải có comparator V4, không chỉ STAIR gốc để tạo cảm giác toàn bộ gain do WMSG.

## 4. Các sai lệch cần sửa trước khi viết paper

| Claim/bảng hiện tại | Đối soát | Hệ quả |
|---|---|---|
| V7 Sports .0520, Electronics .0318 | Ledger selected là .051627377/.031643 | Không dùng làm best-model ranking |
| GĐ4-v1 Electronics .0675/.0310 tại best395 | Đây là epoch500; selected395=.0671/.0309 | Không thay checkpoint để nâng thành tích |
| GĐ4-v3 Baby .1038/.0455 | Epoch500; selected300=.1033/.0448 | Không promote last epoch theo TEST |
| GĐ4-v3 Electronics .0670/.0299 là TEST | VALID450; run dừng451 | Đánh dấu TEST unavailable |
| GĐ4-v5 bảng TEST | JSONL chỉ valid, TEST columns trống | Chưa có TEST để xếp hạng |
| V8 sửa hub bằng inverse-frequency | Source dùng .05+.95[max(0,cos)]² | Sửa mô tả mechanism; chưa có hub-correction evidence |
| GĐ6-v1 Sports operator delta5.27% | Manifest ghi8.0919% | Không copy số Baby sang Sports |
| GĐ6-v2 “adaptive smaller final graph” | Kappa=.25, final support vẫn bằng V8 | Adaptive K chỉ ở auxiliary view |
| KKT/contraction chứng minh tăng ranking | Chỉ chứng minh nghiệm local QP/bounded operator | Không suy ra utility, convergence hay SOTA |
| “Tỷ lệ vàng” graph delta5–6% | Không có định lý/tài liệu xác nhận | Bỏ ngôn ngữ pseudo-theoretical |
| 800.67MiB thỏa <800MiB | Sai số học và scope memory cần phân biệt | Không hứa absolute OOM safety |
| V2 nhanh hơn V1 vì bỏ clustering | Phần offline dưới một giây không giải thích phút giảm trong fit | Profile matched scopes/hardware, không gán nhân quả |

Ví dụ timing: V1 Baby manifest có latent preprocessing≈.866s, category phase≈.660s, total≈28.359s; không xác nhận12.45s mà bảng V2 dùng để suy ra≈39×. Một lần đo khác nhau chưa chứng minh solver làm train nhanh hơn. So sánh mean allocated, peak allocated, reserved và NVML/process phải tách riêng.

Baby historical baseline R@1 là .0123, không phải .0113; một số headline Top1 tăng trưởng bị phóng đại. Không dùng R@1 để thay thế primary N20/R20 khi các số primary gần như ngang nhau.

## 5. Methodology verdict và mức sẵn sàng paper

**Verdict: có nền đáng phát triển, nhưng chưa sẵn sàng tuyên bố một paper phương pháp đã được kiểm định đầy đủ.** Đây là judgment của audit, không phải dự báo quyết định hội nghị.

Những gì đã có: implementation modular, validation-based checkpoint selection, full ranking, static sparse construction, actual runs trên ba dataset, mathematical contracts có thể kiểm thử. Những gì còn thiếu:

1. Ba đến năm seed với matched controls; hiện các run chính là seed1.
2. Ablation tách NLGCL, CSE, WMSG; fixed/shuffled-weight controls. Có thể dùng factorial8 arms: binary/weighted × CF off/on × NLGCL off/on, sau đó chỉ final arms chạy multi-seed đầy đủ.
3. V4/V8 cùng runtime và frozen candidate IDs. Lịch sử CPU/GPU kNN/PyTorch khác nhau làm cho “chỉ thay trọng số” chưa hoàn toàn được chứng minh.
4. Strong external baselines trên cùng split/features/dimension/evaluator: STAIR, LightGCN/MF, BM3, FREEDOM, MGCN, SMORE; thêm DA-MRS/AlignRec nếu mechanism mới nhắm alignment/consensus.
5. Confidence intervals, per-user/degree-stratum metrics, modality sensitivity và resource measurements cùng scope.
6. Novelty against prior work: weighted kNN, co-occurrence và contrastive view đều không mới riêng lẻ. “Áp dụng vào STAIR” không đủ.
7. Disclose adaptive reuse của cùng benchmark TEST qua nhiều version. Chưa thấy bằng chứng TEST chọn epoch, nhưng chọn kiến trúc sau nhiều lần xem TEST vẫn tạo selection risk; freeze quyết định từ validation và thêm confirmation split/dataset nếu có thể.

Một claim có triển vọng hơn: **vai trò của hybrid semantic–behavioral graph khi điều chỉnh update thay vì forward representation, với chi phí inference giữ nguyên**. Cần đối chứng forward-vs-optimizer intervention cùng graph, không chỉ gọi polynomial smoother là “spectral innovation”. Không thể thay việc kiểm định bằng thêm nhiều tên module.

## 6. Literature mới và ý nghĩa cho STAIR

Các nguồn sau đã kiểm chứng title/author/venue hoặc trạng thái preprint. Không nhập gain trong paper khác thành gain kỳ vọng của STAIR; dataset, feature extractor và protocol có thể khác.

| Công trình / tác giả / xuất bản | Cơ chế liên quan và điều rút ra |
|---|---|
| **STAIR: Manipulating Collaborative and Multimodal Information for E-Commerce Recommendation** — Cong Xu, Yunhang He, Jun Wang, Wei Zhang; AAAI2025. [Paper](https://arxiv.org/abs/2412.11729) | Backbone MI/FSC/constrained updates đã là đóng góp của STAIR; phần nghiên cứu mới phải vượt khỏi việc lặp lại cơ chế này |
| **A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation** — Xin Zhou, Zhiqi Shen; ACM MM2023, DOI10.1145/3581783.3611943. [Paper](https://arxiv.org/abs/2211.06924) | Static item graph/degree-sensitive denoising là precedent, không phải novelty mới của V8 |
| **Bootstrap Latent Representations for Multi-modal Recommendation** — Xin Zhou, Hongyu Zhou, Yong Liu, Zhiwei Zeng, Chunyan Miao, Pengwei Wang, Yuan You, Feijun Jiang; WWW2023, DOI10.1145/3543507.3583251. [Paper](https://arxiv.org/abs/2207.05969) | Bootstrap/self-supervision không cần explicit contrastive negatives; benchmark hiệu năng/chi phí quan trọng |
| **Multi-View Graph Convolutional Network for Multimedia Recommendation** — Penghang Yu, Zhiyi Tan, Guanming Lu, Bing-Kun Bao; ACM MM2023, DOI10.1145/3581783.3613915. [Paper](https://arxiv.org/abs/2308.03588) | Behavior-guided modality purification và fusion; một gate dựa hành vi đơn giản không tự tạo novelty |
| **AlignRec: Aligning and Training in Multimodal Recommendations** — Yifan Liu, Kangning Zhang, Xiangyuan Ren, Yanhua Huang, Jiarui Jin, Yingjie Qin, Ruilong Su, Ruiwen Xu, Yong Yu, Weinan Zhang; CIKM2024. [Paper](https://arxiv.org/abs/2403.12384) | Phân lập alignment content–content, content–ID và user–item; làm rõ bottleneck representation trước khi tiếp tục chỉnh graph |
| **Improving Multi-modal Recommender Systems by Denoising and Aligning Multi-modal Content and User Feedback** — Guipeng Xv, Xinyu Li, Ruobing Xie, Chen Lin, Chong Liu, Feng Xia, Zhanhui Kang, Leyu Lin; KDD2024, DOI10.1145/3637528.3671703. [Paper](https://arxiv.org/abs/2406.12501) | DA-MRS dùng cross-modal consistency, denoised feedback và preference/graded-relation alignment; đối thủ gần của consensus proposal |
| **Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation** — Rongqing Kenneth Ong, Andy W. H. Khong; WSDM2025, DOI10.1145/3701551.3703561. [Paper](https://arxiv.org/abs/2412.14978) | Spectral modality fusion/noise treatment; baseline mới hơn cần chạy lại trong protocol chung |
| **A Reproducibility Study of Multimodal Embeddings for Recommender Systems** — Yu Ye, Junchen Fu, Yu Song, Kaiwen Zheng, Joemon M. Jose; International Journal of Multimedia Information Retrieval15, article14,2026, DOI10.1007/s13735-026-00408-9. [Publisher](https://link.springer.com/article/10.1007/s13735-026-00408-9) | Kiểm tra14 models bằng modality knockout; text thường mạnh, thêm image không luôn có lợi. Cần đo modality reliance; perturbation không phải causal proof tuyệt đối |
| **Towards Trustworthy Multimodal Recommendation** — Zixuan Li; arXiv2602.00730,2026 **preprint**, chưa gán venue nhận. [Paper](https://arxiv.org/abs/2602.00730) | Projection/Sinkhorn soft correspondences và corruption tests; không bê dense matching toàn catalog vào Electronics |

Tổng hợp có mục tiêu: literature cho lý do nghiên cứu **representation utility, preference alignment và robustness**, không bảo đảm thêm một semantic gate sẽ tăng metric. Phương pháp spectral mới cũng không tự chứng minh STAIR cần sửa FSC. CoGate manuscript trong repository là nguồn tạo giả thuyết; chưa dùng như một SOTA publication độc lập được xác nhận.

## 7. Hướng tiếp theo: ưu tiên một cơ chế có đường gradient mới

### 7.1 Chẩn đoán trước, không tiếp tục graph tweaking hàng loạt

**OBSERVATION:** V6/v1/v2 đã làm operator thay đổi có đo được nhưng ranking gần V8. Điều này **không chứng minh** graph đã tối ưu hay mọi graph intervention vô ích. Nó cho thấy các proxy hình học/co-interest hiện tại chưa đem lại lợi ích ranking rõ ràng.

**HYPOTHESIS ưu tiên:** bottleneck còn lại nằm ở cách biểu diễn nội dung/hành vi nhận supervision, không chỉ trọng số một graph static dùng trong item-update smoothing.

Trước khi gắn thêm module:

1. **Modality reliance:** permutation item rows của text/image theo train-degree strata, giữ dimension và distribution; chạy retrain controls. Tách MI-only permutation và graph-weight-only intervention để biết đường nào quan trọng. Toàn pipeline permutation là diagnostic tổng, không đủ định vị nguyên nhân.
2. **Không dùng all-zero features trong MI một cách ngây thơ:** centered SVD của input suy biến có thể trả basis tùy ý; thay đổi đó không chỉ là “bỏ semantics”. Log rank/norm/ties và dùng controls tránh artifact này.
3. **Capacity:** đo effective rank, singular spectrum, tail positive-negative margin, learning curves với64/128/256D. STAIR whitening giữ U vectors và rescale; không diễn giải phần trăm raw singular-value energy như phần trăm recommendation information bảo toàn.
4. **Compatibility:** thỉnh thoảng đo cosine của item updates/gradient BPR với gradient auxiliary. Norm gần0 phải đánh dấu unknown, không normalize thành noise.
5. **Contribution:** compare CSE-only, WMSG-only và combined trên frozen candidates; head/tail recall, exposure, modal-weight placebo và matched dose.

Nếu vision permutation không làm giảm ranking, không có cơ sở giải thích gain bằng “fusion image tốt hơn”. Nếu fixed/shuffled-K ngang CAN, không nên bán Adaptive-K là đóng góp chính.

### 7.2 Ưu tiên thực nghiệm: V8.1 Bounded Soft-Consensus View

**Đề xuất:** kiểm định kiến trúc **V8.1 đã có**, thay vì thiết kế ngay thêm V6-v3 hoặc chồng CAN+LCAE+spectral filters. View consensus chỉ tác động một nhánh query của NLGCL; giữ BPR, graph V8, evaluator và inference path.

Schematic theo đặc tả hiện có, không thay thế canonical V8.1:

\[
m_i=(S_{cons}V^0)_i,\qquad
a_i=\min\left(1,\frac{\|I_i^1\|_2}{\|m_i\|_2+\epsilon}\right),\qquad
q_i=I_i^1+\rho_t\,g_i\,a_i\,m_i.
\]

S_cons là graph train-only/content-supported trên support CF đã có; không thêm2-hop expansion. Dùng đúng construction/coverage của `STAIR5_v8_1_Report.md`. g_i∈[0,1] là scalar gate, context từ norms/cosine/coverage; cap/context detached như implementation. Đây là soft evidence, không phải xác nhận kinh tế học rằng item là complements. Không so sánh coordinate trực tiếp giữa modalities chưa align.

**DERIVATION:** với0≤rho_t≤rho_max, forward residual thỏa

\[
\|q_i-I_i^1\|_2\le\rho_{max}\|I_i^1\|_2.
\]

Bound này kiểm soát tỷ lệ chuẩn forward, **không** chứng minh bounded Jacobian mọi tham số, không đảm bảo gradient tương thích với BPR hoặc tăng NDCG. Nếu query norm0 hoặc coverage0 thì neutral fallback theo contract. Gate hiện tại3→8→1 có41 tham số; serving dùng native FSC nên không thêm consensus inference. Chi phí training thêm sparse message/gate vẫn phải đo, đặc biệt256D.

**Causal chain cần kiểm định:** static optimizer graph có tín hiệu train/content → consensus query đưa tín hiệu đó trực tiếp vào contrastive supervision → item gradients thay đổi dù serving graph không đổi → margins/representation quality có thể cải thiện → ranking. V8 không có auxiliary query này. Tuy nhiên CSE đã dùng cùng behavior evidence nên redundancy là rủi ro chính, không được coi gain là tất yếu.

**Tại sao chọn hướng này:** sửa một đường supervision cụ thể, đã có specification/code để pilot, có neutral recovery và norm cap. Nó khác các lần chỉ hiệu chỉnh graph; cùng lúc literature alignment/denoising đã chứng minh đây là family đáng đối chiếu. Novelty component vẫn thấp, đóng góp phải nằm ở bounded intervention, điều kiện có ích và hiệu quả tái lập.

### 7.3 Factorial bắt buộc: capacity × consensus

| Arm | Dimension | Consensus | Câu hỏi |
|---|---:|---|---|
| A | 64 | Off | V8 control |
| B | 256 | Off | Tăng chiều riêng có lợi không? |
| C | 64 | On | Consensus cùng capacity có lợi không? |
| D | 256 | On | Consensus có interaction với dimension không? |

Với một metric m, tính trên paired seeds:

\[
\Delta_{cons,64}=m_C-m_A,\quad
\Delta_{cons,256}=m_D-m_B,\quad
J=\Delta_{cons,256}-\Delta_{cons,64}.
\]

J>0 một lần không chứng minh synergy. Phải có uncertainty. Nếu B tăng còn D≈B, không gọi gain256 là consensus gain. Nếu C>A và D>B, consensus có bằng chứng riêng. Nếu D chỉ thắng A, kết luận vẫn bị capacity confound.

Kèm STAIR/V4 dimension-matched controls ở final stage: tăng256D tự nó không đủ novelty. Dimension intervention cũng thay số U components giữ lại, scale sqrt(N/d), parameter count và discretization của beta schedule; không được nói “chỉ giữ thêm raw variance”. Graph candidate IDs/weights/CSE và loss hyperparameters phải giữ cố định giữa dimension arms theo contract hiện có.

### 7.4 Controls có khả năng bác bỏ consensus

- Uniform/fixed scalar gate, so với learned41-parameter gate: gain có thật do adaptivity không?
- CF-only message, content-conditioned message: content agreement có thêm gì ngoài CSE?
- Degree-stratified shuffled consensus weights/support: proxy có ý nghĩa hay chỉ regularization?
- Bound on/off ở phạm vi nhỏ và profile safe: norm cap có giúp ổn định/tails không?
- Same-dimensional parameter-matched light query adapter: gain có chỉ đến từ extra parameters?
- Ramp và rho grid nhỏ, shared tuning budget; không đồng thời thay LR, negatives, FSC hay CL lambda.

Giữ false-negative modifications tách riêng: KPE trước đó không tăng rõ; không lặp lại một denominator fix rồi gán tất cả vấn đề cho negatives. Không học gate từ TEST hay dùng validation labels làm train supervision trá hình.

## 8. Alternatives và thứ tự ưu tiên

| Hướng | Lợi ích có thể có | Rủi ro / quyết định |
|---|---|---|
| Capacity-only64→128/256 | Kiểm tra bottleneck dimension nhanh nhất về implementation | Có thể overfit/noise, runtime lớn; kiểm định sớm, không coi là novelty chính |
| **Bounded consensus query của V8.1** | Đường supervision mới, neutral recovery, giữ inference | **Primary pilot**; nếu không thắng CF-only/fixed gate thì đơn giản hóa hoặc reject |
| Shared/private hoặc feedback-conditioned alignment | Có thể giải semantic-ID mismatch trực tiếp | Backup sau modality diagnostics; phải so AlignRec/DA-MRS, không thêm MLP rồi gọi rotation-invariant |
| Robust feature/graph treatment | Có câu chuyện clean/noisy/trustworthiness hữu ích cho paper | Dùng controlled corruption; không dùng dense Sinkhorn N×N, không chỉ tăng metric clean |
| CAN+LCAE stacked với extra spectral modules | Nhiều proxy/gate hơn | Không primary: gain trước quá nhỏ, khó attribution, số knobs tăng |
| Price/category substitute–complement/sign gate | Có vẻ giải ambiguity | Chưa có metadata/identified relation labels; không có cơ sở khẳng định sign từ giá/category |
| Dynamic UCR/MAG every-step | Dựa update geometry | Overhead và novelty/robustness burden cao; không ưu tiên vì metric ledger V7 không vượt rõ |

Không có cơ sở gán xác suất “hướng này sẽ tăng5%”. Chọn primary dựa vào khả năng test cơ chế và source sẵn có, không phải một dự báo SOTA.

## 9. Lộ trình thực nghiệm và tiêu chí paper

### Giai đoạn A — sửa evidence ledger và controls

Freeze run IDs, source/runtime hashes, ordered features/splits/candidates. Regenerate selected TEST tables từ artifact, lưu full precision nếu monitor có. Rerun V8 và V4-control trong cùng environment/candidates. Làm modality diagnostics nhỏ; không sửa các historical scores để trông như cùng protocol.

### Giai đoạn B — Baby/Sports pilot

2–5 epoch smoke; sau đó100-epoch paired pilot A/B/C/D để kiểm tra gradients, dose, memory và throughput. Pilot100 không chứng minh final win vì Baby từng chọn430, Sports500. No-go cho NaN, mất neutral parity hoặc resource overrun; early metric lag đơn lẻ chưa đủ bác bỏ.

Dimension256 làm embedding/SpMM/FSC storage và arithmetic tăng theo d; CL logits memory còn phụ thuộc chunks. Không suy ra runtime đúng4× hay cùng8.78h; profile actual. Dùng cùng effective chunk cho paired arms nếu so efficiency, hoặc công khai resource-tuning protocol. Kaggle session limit phải được xử lý bằng checkpoint/resume đã kiểm thử, không hy sinh selection protocol để fit time limit.

### Giai đoạn C — full horizon và confirmation

Chạy đủ native horizon và chọn checkpoint bằng validation N20. Chỉ đưa top candidates sang Electronics sau speed/memory pilot. Tối thiểu3 paired seeds, ưu tiên5 cho final V8/final proposal. Không chạy tất cả knobs trên Electronics ngay. Nếu256D vượt budget,128D là capacity arm mới phải được preregister/đối chứng, không đổi âm thầm sau khi xem TEST.

Report mean±SD, paired absolute/relative differences và confidence intervals; benchmark user-level bootstrap chỉ là sampling uncertainty, không thay thế seed variance. Với5 seed, exact two-sided Wilcoxon có p nhỏ nhất.0625 khi không ties, nên không hứa p<.05 bằng test đó; paired t cần giả định phù hợp, hoặc tăng seed số lượng. Multiple configurations/datasets phải được disclose, không chỉ công bố winning cell.

### Giai đoạn D — mechanism và publication package

Đo query residual ratio, gate/coverage distribution, gradient cosine và norm, BPR/CL margins, effective rank, head/tail recall, exposure, training+evaluation+preprocess separately. Controlled feature row permutations/noise mức nhỏ giúp kiểm tra robustness; inference-only corruption không đủ vì feature đã đi vào training/MI.

**GO:** paired gain tái lập, cùng dimension, không có regression material ở dataset/tail, chi phí chấp nhận được. **WEAK-GO:** mean tăng nhỏ nhưng interval rộng; thêm evidence, chưa claim superiority. **NO-GO:** fixed/CF-only/placebo đạt ngang hoặc tốt hơn và cơ chế dự đoán không xuất hiện. Nếu đạt5% nhưng thiếu fairness/novelty thì paper vẫn yếu; nếu gain thấp hơn nhưng cơ chế, reproducibility và Pareto efficiency tốt thì vẫn có thể có đóng góp đáng nộp.

## 10. Venue fit: rank B được xác minh, không nhầm bảng xếp hạng

Ở đây “B” là **ICORE2026/CORE**, không tự đồng nhất với CCF-B hay Q2 của journal.

| Venue | Rank đã xác minh | Fit và khuyến nghị |
|---|---|---|
| **PAKDD** | **B, ICORE2026**; CORE2023 cũng B, lịch sử2021 A | **Ưu tiên** nếu bài là phương pháp/empirical graph recommendation; cần novelty và evidence thật. [Official ranking](https://portal.core.edu.au/conf-ranks/1667/) |
| **DASFAA** | **B, ICORE2026** | Phù hợp graph/data science/recommendation có scalability và reproducibility tốt. [Official ranking](https://portal.core.edu.au/conf-ranks/988/) |
| **ICTAI** | **B, ICORE2026** | Có thể phù hợp góc algorithmic/AI systems; chọn theo CFP/track thực tế. [Official ranking](https://portal.core.edu.au/conf-ranks/1251/) |

Không dùng trang thứ ba ghi PAKDD A năm2018 để phủ định rank hiện tại. DEXA xuất hiện C trên portal ICORE2026, nên không giới thiệu là CORE-B chỉ dựa một danh sách cũ. Ranking không chứng nhận chất lượng mọi paper và không phải tỷ lệ acceptance.

PAKDD2027 Research Track có recommendation engines, graph GNNs và multimodal learning trong scope. Deadline chính thức hiện là **20/11/2026,23:59 AoE**; double-blind, Springer LNCS. [Research CFP](https://www.pakdd2027.org/pages/calls/research), [Important dates](https://www.pakdd2027.org/pages/dates). Tại ngày audit14/10/2026, thời gian còn ngắn: ưu tiên một hypothesis và confirmation controls thay vì thêm nhiều phiên bản. Đây là gợi ý venue fit có điều kiện, không đăng ký/nộp bài hay tạo lịch tự động.

## 11. Định vị bài báo nên chọn

Nền hiện tại: **V8 WMSG-CSE**. V4 là comparator mạnh; V5.1/V5.2/V6/V7/GĐ6-v1/v2 là evidence về giới hạn/ablations, không phải tất cả đều trở thành module của final model.

Nếu V8.1 thắng đối chứng, story nên tập trung: **norm-controlled consensus supervision bổ sung cho static update smoothing, với inference giữ nguyên và gain không giải thích chỉ bằng dimension**. Nếu chỉ256D thắng, trình bày capacity study/stronger baseline; chưa đủ gọi consensus contribution. Nếu không extension nào thắng, vẫn có thể viết paper về accuracy–efficiency và kiểm định graph interventions, nhưng cần experimental breadth/related-work novelty thật; không dùng các thất bại như bằng chứng rằng V8 là global optimum.

**Khuyến nghị cuối:** chưa bỏ V8 để chạy một stack mới. Trước hết làm sạch ledger và matched controls; tiếp theo chạy factorial **V8-64 / V8-256 / V8.1-64 / V8.1-256**, thêm diagnostics và fixed/CF-only/placebo controls. Đây là hướng có cơ sở để nghiên cứu tiếp, không phải cam kết metric sẽ tăng. Chọn kiến trúc cuối bằng validation và evidence tái lập, rồi mới chốt tên model/paper.
