# STAIR4 v2.1 — Phục hồi baseline và kiểm định regularization trong không gian phức

## 0. Phạm vi, trạng thái và kết luận thiết kế

- Ngày audit: 2026-09-22. Trạng thái: **đặc tả nghiên cứu và kế hoạch triển khai; chưa implement hoặc benchmark v2.1**.
- Áp dụng `academic-research-suite`: nghiên cứu tài liệu có mục tiêu và phản biện phương pháp. Review thực hiện inline, cùng ngữ cảnh; `NOT_CALIBRATED`, không phải hội đồng độc lập hoặc đánh giá acceptance của một venue.
- Đầu vào: `STAIR4_v2_Report.md`, `STAIR4_v2_Experiment_Report.md`, notebook `sta-v2.ipynb`, code và log local, đề xuất QISF/HQCL người dùng cung cấp.
- Các câu như “chạy ngay”, “giải cứu thành công”, “SOTA” trong đề xuất là nội dung được phản biện, không phải kết quả đã quan sát hoặc chỉ dẫn triển khai trong lượt này.
- Mục tiêu: tìm cấu hình cải thiện **validation NDCG@20**, xác nhận bằng test tại checkpoint được chọn, đồng thời báo Recall@20 và chi phí. Không có bảo đảm toán học rằng cấu hình mới vượt baseline.

**Kết luận chính:** chưa nên thay BSC bằng QISF. Code v2 có sai lệch chuẩn hóa FSC so với baseline, xuất hiện ngay cả khi tắt mọi module mới. Phải tách việc sửa lỗi tương đương baseline khỏi đóng góp kiến trúc. Sau đó mới kiểm định một auxiliary loss được định nghĩa nhất quán, có đối chứng cosine và fidelity. Giữ BSC gốc trong v2.1 chính; bộ lọc residual chỉ là nhánh thử nghiệm thứ hai.

Tên làm việc: **STAIR4-v2.1: Baseline-preserving Complex Cross-layer Regularization (BCCR)**. “Complex” mô tả phép tính; không hàm ý quantum advantage hoặc quantum entanglement. Nếu biến thể cosine thắng, chọn cosine và sửa tên khi báo cáo, không giữ câu chuyện lượng tử bằng mọi giá.

## 1. Hồ sơ bằng chứng và khả năng truy nguyên

### 1.1. Tài liệu đã đối chiếu

| Artifact | Vai trò | Giới hạn |
|---|---|---|
| `docs/giai_doan_4/STAIR4_v2_Report.md` | Đặc tả ban đầu, baseline contract, công thức BSF/POCL | Là thiết kế; không phải chứng cứ thực nghiệm |
| `docs/giai_doan_4/STAIR4_v2_Experiment_Report.md` | Diễn giải kết quả và giả thuyết nguyên nhân | Có kết luận nhân quả vượt dữ liệu và dùng simulation như telemetry |
| `notebook/P4/sta-v2.ipynb` | 42 cells, pipeline và hình phân tích | Một số hình chủ động gắn nhãn analytical/simulation |
| `models/stair4_v2.py:480–497`; `main.py:202–207` | Đối chiếu FSC | Xác nhận lỗi ở source hiện tại và commit ghi trong manifest |
| `main_stair4_v2.py:325–332` | Monitor `LOSS` | Giá trị được ghi là total loss |
| Hai `log.txt` và `run_manifest.json` dưới `logs/GD4/` | Kết quả run, epoch chọn, cấu hình | Manifest có `git_dirty=true`, không lưu đầy đủ diff |

Đường dẫn log gốc:

```text
logs/GD4/STAIR4-v2/Amazon2014Baby_550_MMRec/0922074606/log.txt
logs/GD4/STAIR4-v2/Amazon2014Sports_550_MMRec/0922083722/log.txt
logs/GD4/stair4v2_Amazon2014Baby_550_MMRec_A6_20260922_074630/run_manifest.json
logs/GD4/stair4v2_Amazon2014Sports_550_MMRec_A6_20260922_083810/run_manifest.json
```

Hai manifest đều ghi commit `ab2f7647842c20609443afa320712c103e694700`, A6, `phase_fidelity`, `learned_givens`, `bsf_mix`, warmup 10, ramp 20, auxiliary weight 0.001, temperature 0.2, spectral time 0.5, spectral mix 0.1. Source của commit này cũng có lỗi FSC nêu ở §2. `git_dirty=true` không tự chứng minh code bị sửa: có thể chỉ do untracked files; nhưng thiếu diff/file hashes nên chưa thể khẳng định bit-for-bit source thực chạy.

SHA256 của ba đầu vào tại thời điểm audit:

```text
sta-v2.ipynb:
3FD58D809DECD3B0D137AE792D92F46FFE94439FEA94566292D1657136A464FD
STAIR4_v2_Report.md:
FC701BCA6C971ABE9C848026D9212E0221988DF23F4FE9A88F92B74CD8C9CEE9
STAIR4_v2_Experiment_Report.md:
EFC7C27391796BA8B1E52D1E4CE5229E03A929C5A0850F3DF70667470639CB00
```

### 1.2. Kết quả nào thực sự được xác nhận?

| Dataset | Epoch checkpoint chọn bởi validation | Test R@20 trong log | Test N@20 trong log | Baseline được báo cáo R/N@20 | Chênh lệch tương đối R/N |
|---|---:|---:|---:|---|---|
| Baby | 340 | 0.0606 | 0.0255 | 0.1042 / 0.0454 | −41.84% / −43.83% |
| Sports | 490 | 0.0749 | 0.0331 | 0.1111 / 0.0500 | −32.58% / −33.80% |

Các tỷ lệ tính từ số đã làm tròn. Cột baseline lấy từ báo cáo người dùng; chưa phải baseline tái chạy có cùng data/runtime/seed trong audit này. Không trộn test của epoch cuối với test của checkpoint tốt nhất. Không suy từ hai dataset sang “Electronics chắc chắn thất bại”.

`LOSS` của engine là `BPR + lambda * auxiliary`, nên đường LOSS không tự chứng minh BPR plateau ở đúng giá trị đó. Sau warmup cần log riêng từng thành phần. Giá trị thấp hơn của BPR cũng không đủ chứng minh ranking tốt hơn.

## 2. Phát hiện ưu tiên: FSC của v2 chưa phục hồi baseline

Đặt \(b_j=0.1+0.9(j/d)^\gamma\), \(a_j=1-b_j\), \(F^{(0)}=E\), \(F^{(l+1)}=AF^{(l)}\operatorname{diag}(a)\). Baseline dùng:

\[
Z_{:,j}=\frac{b_j}{1-a_j^{L+1}}\sum_{l=0}^{L}F^{(l)}_{:,j}.
\]

Trong v2, `self.beta` chính là \(b\), nhưng dòng 497 nhân `beta_complement`, tức \(a\). Với cùng E và A:

\[
Z^{\mathrm{v2}}_{:,j}=\frac{1-b_j}{b_j}Z^{\mathrm{baseline}}_{:,j}.
\]

Đây là **đổi trọng số theo chiều**, không phải chỉ nhân một hằng số toàn cục. Dot product đổi trọng số đóng góp của chiều j theo bình phương tỷ lệ này. Ví dụ d=64, Baby gamma=0.1:

- j=0: b=0.1, gain embedding bằng 9, gain đóng góp score bằng 81.
- j=63: b≈0.9985837633, gain embedding≈0.0014182452, gain score≈0.0000020114.

Sai lệch tồn tại trước khi auxiliary và BSF được bật. Tại graph mode A có eigenvalue 1, hệ số FSC baseline bằng 1, trong khi v2 bằng a/b. Đây là counterexample ngắn để bắt lỗi bằng unit test.

**Độ tin cậy:** cao về sai lệch source; chưa xác định bao nhiêu phần trăm suy giảm thực nghiệm do lỗi này. Sửa FSC là **bug fix**, không được tính là đóng góp quantum hoặc cải tiến thuật toán. Cần run đối chứng chỉ sửa FSC, giữ nguyên mọi option A6, để định lượng tác động.

### 2.1. Các kết luận post-mortem cần hạ mức chắc chắn

| Nhận định cũ | Phản biện và bằng chứng | Phép đo cần bổ sung |
|---|---|---|
| BSF xén cứng phổ vào [0.05, 0.40] | Không đúng code được audit: H=(I−S)/2, Q=I−t²H²/2, rồi mixture với BSC; không thấy phép clipping như mô tả | Dense eigentest trên graph nhỏ; response trên phổ S thực |
| Fidelity không âm nên không đẩy negatives | Sai về loss: với negative, đạo hàm CE theo logit là p_j>0; gradient descent vẫn giảm similarity. Kernel có thể bão hòa/đạo hàm yếu, nhưng nguyên nhân khác | Phân bố overlap và gradient kernel |
| arctan chắc chắn đã bão hòa | Có thể khi đầu vào lớn; chưa có histogram activation hoặc đạo hàm để xác nhận | Quantiles của input và 1/(1+x²) |
| Góc Givens kẹt <0.15rad | Notebook cell index 36 tạo `theta_traj` bằng `np.random` và gắn nhãn illustrative simulation | Theta/grad/update từ checkpoint thật |
| Detach triệt tiêu mọi xung đột | Chỉ bỏ gradient của target path; query embedding vẫn nhận BPR và CL | Gradient cosine trên shared embeddings |
| VRAM phẳng chứng minh không có peak | NVML sampling và memory allocated không cùng khái niệm; sampling có thể bỏ lỡ peak | Peak allocated/reserved và đo cả backward/step |

Góc nhỏ có thể là nghiệm hợp lý hoặc do learning rate, không phải tự động là “ma sát tối ưu”. Hai thành phần raw gradient cộng tuyến tính; hai bước Adam chạy riêng không cộng tương đương vì moments và mẫu số phụ thuộc gradient tổng.

## 3. Kiểm chứng ba công trình tham khảo

Đây là targeted literature check, không phải systematic review hoặc chứng minh novelty. Trích dẫn đúng phiên bản; không suy từ uy tín venue sang hiệu quả trên STAIR.

| Nguồn | Điều xác nhận được | Điều không được chuyển thành kết luận của v2.1 |
|---|---|---|
| Li & Casiraghi, WWW 2026 [R1] | Tên bài, tác giả, proceedings và DOI khớp metadata Aalto; nghiên cứu representation/matching cho recommendation | Chưa xác nhận từ toàn văn rằng công thức L2Norm+i·tanh do bài này đề xuất; không chứng minh lợi thế ở d=64 trong code này |
| Debi & Makmal, Scientific Reports 2025 [R2] | VQRS dùng MF latent vectors, angle encoding và CNOT. Ablation bỏ entangling layers xuống dưới RAND ở thí nghiệm 32 users/32 items, 5 qubits, 10 iterations | Kết quả cụ thể đó không chứng minh mọi Givens rotation là entanglement hoặc sẽ giúp Amazon full ranking |
| Wang et al., IEEE TETCI 2026 [R3] | Journal 10(1), 830–843; online 2025, DOI 10.1109/TETCI.2025.3604791. ArXiv [R4] là phiên bản riêng | Graph representation trên nền quantum không chứng minh lấy real(Taylor Hamiltonian) cải thiện BSC hoặc recommendation |

[R1: hồ sơ Aalto](https://research.aalto.fi/en/publications/quantum-enhanced-representation-learning-and-matching-learning-fo/); [R2: publisher và Figure 8](https://www.nature.com/articles/s41598-025-15869-x); [R3: metadata CityUHK](https://scholars.cityu.edu.hk/en/publications/qghnn-a-quantum-graph-hamiltonian-neural-network/).

Mức chứng cứ: các nghiên cứu thuật toán đơn lẻ, không phải tổng hợp bằng chứng về STAIR. Không áp dụng máy móc phân cấp RCT y học cho benchmark ML. Metadata nguồn trường/publisher xác nhận sự tồn tại; kiểm chứng nội dung R1 và R3 còn giới hạn ở metadata/abstract, R2 có nội dung phương pháp và ablation. Không thực hiện kiểm tra COI/retraction toàn diện hoặc kiểm toán peer review; không gắn nhãn “SOTA hàng đầu” chỉ dựa vào ba nguồn này.

## 4. Phản biện toán học và code QISF/HQCL được cung cấp

### 4.1. Unitary không đồng nghĩa real smoother bảo toàn chuẩn

Với H thực đối xứng và g thực:

\[
\|e^{-itH}g\|_2=\|g\|_2,\qquad
\Re(e^{-itH}g)=\cos(tH)g.
\]

Lấy H=[1], g=[1], t=0.12: unitary phức có norm 1, phần thực có norm cos(0.12)≈0.992808636, Taylor thực có norm 0.9928. Taylor phức \(1-it-t^2/2\) có norm bình phương \(1+t^4/4>1\). Vì vậy cả “bảo toàn 100%” lẫn “zero spectral distortion” đều sai với phép tính đề xuất.

Nếu H=S và phổ S trong [−1,1], gain \(1-t^2\lambda^2/2\) giảm cả mode λ≈+1 và λ≈−1; không phải low-pass theo Laplacian frequency 1−λ. Nếu dùng H=(I−S)/2, nó trở thành một bộ lọc đa thức low-pass có điều kiện — về bản chất gần BSF v2 đang có, không phải thay thế quantum hoàn toàn mới. Hermitian không đòi PSD; ký hiệu `SPSD(mAdj)` chưa định nghĩa cách xây, độ phức tạp và tác động lên graph.

### 4.2. Encoding và similarity không khớp tuyên bố

1. L2 normalization có Jacobian phụ thuộc norm và mất hướng radial; tanh vẫn bão hòa. Không có “bảo toàn gradient tuyến tính”.
2. Công thức chuẩn hóa toàn vector khác code `z/(abs(z)+eps)` chuẩn hóa từng tọa độ. Code tạo norm gần sqrt(d), không phải 1.
3. Nếu sửa đúng norm=1, còn chia similarity cho d thì tại d=64, temperature=0.2, biên logit chỉ khoảng ±0.078125; softmax dễ gần đều.
4. \(\Re\langle z,w\rangle\) là inner product của hai vector thực ghép real/imag. Nó không phải Fubini–Study distance, vốn dùng \(\arccos|\langle z,w\rangle|\) cho unit states.
5. Givens trên cặp tọa độ không tự chứng minh entanglement. Muốn dùng thuật ngữ đó phải định nghĩa tensor-product subsystems và kiểm chứng separability. Cùng một unitary áp dụng lên cả hai nhánh còn giữ nguyên inner product.

### 4.3. Các lỗi code và tích hợp cần loại bỏ

| Lỗi trong đoạn code đề xuất | Hậu quả | Contract thay thế |
|---|---|---|
| List `layer_embeds` dùng `.device` và `torch.split` trực tiếp | Runtime error; không lấy được X0/X1 | Truyền hai tensor được đặt tên rõ ràng |
| U0/I0 và U1/I1 split cùng đối tượng | Không có cross-layer view thật | X0=E; X1=A@E trước attenuation |
| `entangler(psi).detach()` ở tất cả đường dùng theta | Theta không nhận gradient | `G(encode(X1).detach())` nếu muốn học G |
| `masked_fill` dùng tensor ma trận làm value | Sai scalar-fill API; thao tác diagonal không cần thiết | Tạo positive targets từ train index |
| Broadcast [B,1,d] × [1,B,d] | Materialize tensor B²d: complex64 B=4096,d=64 là 8 GiB cho một tensor | GEMM real/imag và chunk query |
| Diagonal-only positives | Duplicate IDs và train-positive khác bị coi negative | Unique candidates và multi-positive mask |
| Ramp từ epoch đầu | Không phải warmup zero-aux | Warmup và ramp riêng |
| Theta khởi tạo random, không bound | Khác identity-at-init và miền góc đã tuyên bố | Identity init; bound tường minh nếu dùng |
| Xavier, gamma=0.2 cố định, thêm L2 trên Z | Thay MI/FSC/regularization baseline | Kế thừa từng dataset và optimizer decay |
| `apply_qisf_smoothing` không được nối optimizer | Smoother có thể không chạy | Callback trên Adam direction, có integration test |
| Graph buffers None, thiếu prepare/evaluation contract | Không phải model chạy end-to-end | Kế thừa runtime baseline và kiểm chứng ranking |

Không đưa đoạn code này vào pipeline như “production-ready”. Các mốc BPR<0.05, Recall=0.1065 và thời gian 15 phút là dự báo chưa được đo; không dùng làm kết quả hoặc điều kiện pass.

## 5. Câu hỏi nghiên cứu và giả thuyết có thể bác bỏ

**RQ:** sau khi phục hồi chính xác MI/FSC/BSC và evaluation, regularization giữa embedding gốc và first-hop train-only có cải thiện NDCG@20 so với STAIR trong cùng ngân sách không; phần hình học phức đóng góp thêm gì so với cosine?

| Giả thuyết | Dự đoán có thể đo | Khi nào bác bỏ/hạ mức kết luận |
|---|---|---|
| H0-code: FSC sai làm sai lệch A0 | A0 hiện tại khác baseline; A0 sửa FSC khớp forward/step | Còn lệch sau sửa: tìm confound tiếp, chưa chạy architecture search |
| H1: CL bổ sung regularization hữu ích | Corrected backbone + CL cải thiện paired-seed validation NDCG | Không vượt corrected baseline trong budget đã đăng ký |
| H2: fidelity mixture có ích hơn signed overlap/cosine | Gain còn tồn tại khi khớp candidate/loss/schedule/parameter budget | Cosine bằng hoặc tốt hơn: bỏ diễn giải quantum |
| H3: target stop-gradient giúp tối ưu | Gradient diagnostics và ranking tốt hơn nhánh không stopgrad | Chỉ giảm thời gian hoặc không gain: báo đúng vai trò |
| H4: residual spectral correction có ích | Gain vượt cả BSC và identity-mix control | Chỉ tương đương giảm smoothing: không nhận đóng góp spectral riêng |

Tính mới của combination chưa được xác lập bởi tra cứu ba bài; luận văn có thể đóng góp audit, thực nghiệm có kiểm soát và kết quả âm, không cần tự gán quantum advantage.

## 6. Kiến trúc v2.1 chính: corrected STAIR + BCCR

```text
Train interactions + original multimodal features
                 |
       Original MI and static graphs A, S
                 |
          E = [User; Item]
            /           \
Correct FSC polynomial   X0 = E; X1 = A @ E
         |                 |             |
 Z -> dot product BPR   query encoder   stop-gradient -> key rotation
         |                 \             /
         |             unique candidates + train-positive targets
         |                            |
         +--------------------- lambda(epoch) * CL
                                      |
                       AdamWSEvo parameter groups
              users: none; items: original BSC; head: none
                                      |
                  Original full/pool ranking and selection
```

### 6.1. Backbone và inference contract

- Giữ MI bằng modality SVD/user aggregation của baseline, cùng seed và preprocessing. Không thay bằng Xavier.
- FSC dùng đúng công thức §2; không đưa head vào inference. BPR giữ nguyên batch, duplicate weighting và reduction của baseline.
- X1 là `A @ E` **trước** nhân a theo chiều; tái sử dụng intermediate của bước đầu. Không chia ngược tensor đã nhân a vì một số a rất nhỏ.
- BSC vẫn là polynomial Neumann hữu hạn:

\[
P_j(S)\Delta_{:,j}=\frac{\sum_{l=0}^{L} b_j^l S^l\Delta_{:,j}}{\sum_{l=0}^{L}b_j^l}.
\]

- Delta là hướng Adam sau moments và bias correction, không phải raw BPR gradient. Không thêm regularization vào Z nếu baseline chỉ dùng optimizer decay.
- Khi lambda=0, không chạy auxiliary hoặc tiêu thụ RNG phụ; model phải phục hồi baseline từ initialization tới optimizer update. Chỉ kiểm tra `zeta=0` ở smoother là chưa đủ.
- Giữ nguyên full/pool ranking, seen-item mask, validation NDCG@20, cách xử lý ties và checkpoint selection của cùng FreeRec runtime. Test không được dùng chọn model.

### 6.2. Encoder chính: bounded-input phase features

Không lấy công thức tanh của đề xuất làm mặc định. Với x thuộc R^d:

\[
r(x)=\frac{x}{\max(\|x\|_2,\epsilon)},\qquad
\psi_k(x)=\frac{1}{\sqrt d}\exp\{i\kappa\sqrt d\,r_k(x)\}.
\]

Pilot: epsilon=1e−8 trong FP32, kappa=0.5 cố định; kiểm tra thêm kappa=0.25 và 1.0 chỉ trong ngân sách tuning. Đây là lựa chọn thiết kế mới của tài liệu này, **không gán cho R1**.

Ưu điểm có điều kiện: norm ψ=1; loại dependence của angle lên scale tổng của x; không có đạo hàm arctan/tanh bão hòa theo độ lớn x. Hạn chế: normalize vẫn có null direction radial, nhạy near-zero; phase có periodic aliasing, coordinate dependence và có thể collapse. Không tuyên bố “không có vanishing gradient”. Log tỷ lệ norm dưới epsilon; bỏ các cặp chứa view gần zero khỏi auxiliary, giữ chúng trong BPR.

Đối chứng C-real dùng r(x) với cùng candidate/loss. Backend real/imag của ψ phải tương đương complex numerically; nó là kiểm tra triển khai, không phải baseline thuật toán độc lập.

### 6.3. Key-only rotation và vị trí stop-gradient

Với d chẵn, G_theta gồm d/2 blocks quay hai chiều. Primary pilot **identity rotation**; learned rotation chỉ được bật sau ablation. Khi bật:

\[
\theta_k=(\pi/4)\tanh\omega_k,\quad \omega_k(0)=0,\quad
q=\psi(X_0),\quad k=G_\theta\big(\operatorname{sg}(\psi(X_1))\big).
\]

Một G dùng chung hai hướng user→item và item→user. Stop-gradient trước G bảo đảm omega vẫn học. Không detach sau G. Key state được tính mới từ pre-step embeddings ở từng minibatch; không cache autograd graph qua epoch. Đây là stop-gradient online target, **không phải EMA teacher**; EMA là nghiên cứu khác và không có trong v2.1 chính.

Cosine control dùng rotation trực tiếp trên d tọa độ thực với d/2 góc. Code cosine v2 ghép thành d/2 kênh phức rồi dùng Givens chỉ tạo d/4 góc; shape check không chứng minh capacity matching. Đối chứng mới phải đếm trainable scalars và kiểm tra gradient thật.

### 6.4. Similarity hỗn hợp có đối chứng rõ ràng

\[
h(q,k)=q^Hk,\quad s_\eta(q,k)=(1-\eta)\Re h+\eta|h|^2,\quad
\ell=s_\eta/T.
\]

Primary candidate: eta=0.25, T=0.2; fixed hyperparameters, chưa phải optimum. Không chia thêm cho d. Eta=0 là signed complex overlap; eta=1 là fidelity v2 với encoder mới. Do norm=1, |h|≤1 và s nằm trong [−(1−eta),1]. Thành phần fidelity mang tính bất biến global phase; signed component không có tính đó, vì vậy **toàn kernel không phải Fubini–Study**.

Lý do thử: signed component có thể tránh vùng đạo hàm bằng zero của |h|² khi h≈0; fidelity giữ một thành phần tương tác bình phương. Đây là hypothesis, không là bằng chứng hiệu quả. Realification biểu diễn chính xác được cả hai; không có lợi thế biểu diễn chỉ vì dtype complex.

Tính bằng GEMM với q=a+ib, k=c+ie:

\[
\Re h=ac^T+be^T,\qquad \Im h=ae^T-bc^T.
\]

Không materialize [Bq,Bk,d]. Với phase cosine đã gần h=1 cho mọi cặp, phải đo logit variance/entropy để phát hiện collapse, không tăng lambda mù quáng.

### 6.5. Multi-positive CE train-only

Lấy unique users U_B và unique positive items I_B từ batch gốc. Mask M_ui=1 nếu (u,i) thuộc **train interactions**, không chỉ cặp trên diagonal. Không sửa sampler hoặc weighting của BPR.

\[
Y_{ui}=M_{ui}/\sum_jM_{uj},\qquad
L_{u\to i}=-\frac{1}{|V_u|}\sum_{u\in V_u}\sum_iY_{ui}\log\operatorname{softmax}(\ell_{u,:})_i.
\]

V_u chỉ chứa hàng có ít nhất một positive và một candidate không positive, đồng thời query/key views hợp lệ. Hướng ngược dùng mask chuyển vị với logits được tính từ item X0 và user X1, không chỉ chuyển vị logits forward. Nếu một hướng không có hàng hợp lệ thì chỉ dùng hướng còn lại; cả hai trống thì CL=0 và ghi diagnostic.

\[
L_{CL}=\tfrac12(L_{u\to i}+L_{i\to u}),\qquad L=L_{BPR}+\lambda(e)L_{CL}.
\]

Hệ số 1/2 chỉ dùng khi cả hai hướng hợp lệ. Uniform targets giữ gần thiết kế v2 và tránh false-negative với known positives; nó vẫn có thể ép các positives có độ tin cậy khác nhau về xác suất đều. Unknown positives vẫn có thể bị xem là negative. So sánh candidate distributions giữa các run; không gọi biện pháp này là loại bỏ hoàn toàn false negatives.

### 6.6. Schedule và kiểm soát gradient

Dùng epoch index e bắt đầu từ 0:

\[
r(e)=\operatorname{clip}((e-W)/R,0,1),\qquad\lambda(e)=\lambda_{\max}r(e).
\]

Pilot: W=10, R=20 để có đối chứng với manifest v2; lambda_max=1e−4, đối chiếu 1e−3. Không mặc định 50 epoch hoặc claim các mức này tối ưu. Nếu muốn đổi W thì phải đăng ký một ablation riêng. Không dùng validation feedback để tự đổi lambda giữa run mà không ghi đây là algorithm khác.

Tại diagnostic steps, tính raw gradients BPR/CL bằng `autograd.grad` trước backward chính, không ghi đè `.grad` và không optimizer step giữa hai phép đo:

\[
c=\frac{\langle g_B,g_C\rangle}{\|g_B\|\|g_C\|+\epsilon},\qquad
\rho_g=\frac{\lambda\|g_C\|}{\|g_B\|+\epsilon}.
\]

Đo riêng users/items và trên cùng tọa độ; ghi zero-norm flags. Sau backward mới ghi grad của omega, trước step lưu theta, sau step ghi delta-theta. Pilot log mỗi 100 steps; overhead được báo riêng. Không thêm PCGrad/gradient clipping/adaptive controller vào primary để tránh thêm confound. c<0 là dấu hiệu xung đột cục bộ, không chứng minh CL làm giảm NDCG; c≥0 cũng không bảo đảm generalization.

## 7. Nhánh tùy chọn v2.1-S: residual spectral correction có giới hạn

Chỉ mở sau khi v2.1 chính và baseline đã được xác nhận. Không gọi nó là unitary smoother.

Giả sử S thực đối xứng, nonnegative normalized adjacency, phổ trong [−1,1]. Đặt H=(I−S)/2; p=P(S)Delta. Dùng:

\[
T_\xi(\Delta)=(I-\xi H^2)p,\qquad 0\le\xi\le0.1.
\]

Đây là **composition sau BSC**, khác mixture v2 `(1−zeta)PDelta+zeta QDelta`. Xi=0 trả nguyên p qua fast path. Gain bổ sung trong [1−xi,1], giữ mode λ(S)=1 và giảm mode tần số cao. Có bound:

\[
\|T_\xi(\Delta)-P\Delta\|_F\le\xi\|P\Delta\|_F.
\]

Bound chỉ giới hạn perturbation hướng update khi giả thiết phổ đúng; không chứng minh descent, không bảo toàn norm và không suy ra gain ranking. Chi phí hai SpMM phụ; không xây H² hoặc matrix exponential. Smoother không có parameter, nhận scalar xi đã detached tại pre-step, clear state trong finally.

Pilot xi_max∈{0.025,0.05}; lịch ramp như CL. Đối chứng bắt buộc: BSC nguyên bản, identity-mix `(1−xi)PDelta+xi Delta`, và BSF mix v2 trên corrected backbone. Nếu S không đối xứng/chuẩn hóa đúng, bound không dùng được; không âm thầm symmetrize graph baseline chỉ để đạt chứng minh.

Rủi ro chính: BSC đã smoothing; thêm H² có thể oversmooth và hại tail items. Vì vậy xi=0 là mặc định trong kiến trúc chính. Không ghép nhiều thay đổi cùng lúc rồi quy công cho spectral design.

## 8. Optimizer, autograd và bộ nhớ

| Group | Optimizer/rate | Smoother | Decay |
|---|---|---|---|
| User embeddings | AdamWSEvo, baseline lr | None | Baseline |
| Item embeddings | AdamWSEvo, baseline lr | Original BSC; optional T_xi | Baseline |
| Rotation omega nếu bật | AdamWSEvo group, 0.1×lr | None | 0 |

Mỗi parameter có đúng một group; identity head không có trainable parameter thì không tạo empty optimizer giả. Giữ nguyên optimizer epsilon/betas/update order. Init auxiliary không được làm đổi RNG dùng cho MI/sampling; lưu RNG-state hoặc dùng generator riêng.

`prepare()` chỉ lưu static tensors không có grad_fn: MI, A, S, train-positive index, candidate-independent constants. X0/X1 và auxiliary weights được tính mới mỗi step. Snapshot nếu dùng nhánh spectral phải có trước mọi group step, detached, clear trong finally; không recompute từ embeddings sau khi một group đã cập nhật.

Memory: phase encoding O((Bq+Bk)d); similarity O(BqBk), không O(BqBkd). Bq=Bk=4096: một float32 matrix là 64 MiB; còn masks, intermediates và saved tensors. Chunk query size=256 là pilot, không phải bảo đảm VRAM. **Chỉ chia forward thành chunks rồi cộng loss có thể vẫn giữ toàn bộ saved autograd tensors**; muốn peak giảm thực sự phải dùng recomputation/checkpoint hoặc backward theo chunk với scaling chính xác và không step giữa chunks. Verify dense/chunked gradient parity.

Primary auxiliary FP32 trên Tesla T4; không mặc định BF16/complex AMP. Benchmark cả chuẩn bị graph, epoch train và full-ranking evaluation. Reset/ghi CUDA peak allocated/reserved theo phase, cùng NVML process memory nếu có. Không ghi “VRAM phẳng tuyệt đối”.

## 9. Kế hoạch thực nghiệm: tách bug fix khỏi đóng góp

### 9.1. Gate 0 — tương đương baseline trước GPU dài hạn

1. Cùng static tensors E/A/S, so baseline và v2.1 lambda=xi=0: Z, scores, BPR, gradients, một và nhiều optimizer steps, moments, decay.
2. Synthetic graph có mode λ=1 phải bắt được lỗi a/b; d≥4, beta khác nhau theo chiều, không dùng beta=0.5 vì lỗi sẽ bị che.
3. So top-k và seen-item masking full/pool trên toy split. Tie handling và checkpoint selection phải khớp.
4. Resume checkpoint phải khớp uninterrupted run khi cùng RNG/sampler state và môi trường deterministic.

Pilot tolerance FP32 cho phép toán nhỏ: atol=1e−6, rtol=1e−5; báo max error và sparse backend. Không nâng tolerance để che sai công thức. CPU float64 oracle dùng để phân biệt roundoff với bug. Pass unit test shape hoặc một epoch trước warmup không thay thế Gate 0.

### 9.2. Ma trận ablation theo giai đoạn

| ID | Backbone | CL | Smoother | Câu hỏi |
|---|---|---|---|---|
| B0 | Baseline main.py | Off | BSC | Baseline tái lập |
| B1 | Corrected v2.1 | Off | BSC | Đã phục hồi baseline chưa? |
| F0 | v2 với FSC cũ, cùng config A6 | POCL cũ | BSF cũ | Reproduce failure nếu cần; không gọi baseline |
| F1 | Chỉ sửa FSC trên v2 A6 | POCL cũ | BSF cũ | Tác động riêng của bug fix |
| C0 | Corrected | Cosine real, identity, stopgrad | BSC | CL đơn giản có đủ? |
| C1 | Corrected | Signed phase eta=0 | BSC | Phase mapping so với cosine |
| C2 | Corrected | Mixture eta=0.25 | BSC | v2.1 chính có gain? |
| C3 | Corrected | Fidelity eta=1 | BSC | Squared overlap có ích riêng? |
| C4 | Corrected | C2 + learned rotation | BSC | Rotation thêm ích gì? |
| C5 | Corrected | C2, target không stopgrad | BSC | Vai trò stopgrad |
| S0/S1 | Corrected | Off / CL đã chọn | Residual spectral | Spectral độc lập/tương tác |

Không chạy full Cartesian grid. Stage 1 Baby: B0/B1/F1 và C0/C1/C2/C3 với seed=1, pilot 100 epochs, chỉ để phát hiện lỗi và xếp ưu tiên. Stage 2: tối đa ba cấu hình được chọn trước trên validation, bao gồm B0, chạy **đủ budget baseline** trên Baby và Sports, paired seeds {1,2,3,4,5}. Nếu tài nguyên chỉ đủ ba seeds, ghi exploratory và hạn chế CI. Stage 3 Electronics dùng recipe đã khóa hoặc tuning budget ngang baseline; không sửa recipe dựa trên test.

Candidate search budget ban đầu: C0 và C2 mỗi loại lambda∈{1e−4,1e−3}, giữ T=0.2, W/R=10/20; C1/C3 dùng lambda đã chọn theo quy tắc validation đăng ký trước. Điều này là staged search, không phải so sánh hyperparameter tối ưu toàn cục. Rotation và spectral chỉ được mở khi core có tín hiệu; ghi tổng số trials của mọi nhánh, bao gồm thất bại.

### 9.3. Tiêu chí quyết định

- Primary metric: test NDCG@20 tại checkpoint được chọn **chỉ bằng validation NDCG@20**. Recall@20 là secondary được báo đầy đủ, không đổi primary sau khi thấy kết quả.
- Báo từng seed, mean/std, paired differences và CI với cách tính nêu rõ; năm seeds vẫn cho CI bất định. User bootstrap đo biến thiên users có điều kiện trên model, không thay thế training-seed uncertainty.
- Chỉ tuyên bố “vượt baseline trên dataset X” khi paired comparison và mức gain thực tế hỗ trợ kết luận; nếu CI chứa zero, ghi chưa đủ bằng chứng. Nếu nhiều so sánh inferential, điều chỉnh multiplicity hoặc chỉ coi một cấu hình khóa trước là confirmatory.
- Không đặt ngưỡng BPR loss để tuyên bố hồi phục. Không so baseline 500 epochs với v2.1 100 epochs rồi kết luận cuối cùng; learning-curve và budget phải được trình bày rõ.
- Pilot có NaN, memory failure, gradient theta=None ngoài identity mode, A0 mismatch hoặc leakage: dừng để sửa. Pilot không tăng metric là kết quả hợp lệ, không xóa run.
- Phân tích thêm head/mid/tail theo **train degree**, margin positive-negative và MI drift. Không chọn subgroup sau khi xem test để quảng cáo gain.

## 10. Diagnostics và provenance cần có

Mỗi run lưu immutable config, command, code SHA, git diff nếu dirty, hash các source quan trọng, data split/features/cache hashes, environment versions, seed, GPU, precision, timestamps, stop reason và metric-selection rule. Notebook dùng source ở commit cố định; không `reset --hard origin/main` giữa chuỗi experiment.

Log JSONL tối thiểu:

```text
epoch, step, bpr_raw, cl_raw, cl_weighted, total_loss, lambda, xi
candidate_users, candidate_items, valid_rows_u/i, skipped_rows_u/i
logit_mean/std, softmax_entropy, positive_margin, near_zero_view_rate
grad_bpr_norm_u/i, grad_cl_norm_u/i, grad_cos_u/i, weighted_grad_ratio_u/i
theta_norm, theta_grad_norm, theta_update_norm
embedding_norm_quantiles, fsc_channel_gain_checks, mi_drift
step_ms, peak_allocated_mib, peak_reserved_mib, nvml_process_mib
```

Diagnostics phải detached/Python scalars; không giữ graph trong list history. Các hình analytical/simulation nằm riêng và gắn nhãn; chỉ telemetry có run_id/step được dùng làm bằng chứng tối ưu. Snapshot theta từ model trước/sau optimizer.step mới được vẽ trajectory học thật.

## 11. Kế hoạch file mã nguồn — chưa triển khai trong tài liệu này

| File đề xuất | Trách nhiệm và điều kiện hoàn thành |
|---|---|
| `models/stair4_v2_1.py` | Backbone corrected FSC, reuse MI/graphs/scorer; X0/X1 rõ nghĩa; forward bỏ hẳn CL khi lambda=0; diagnostics detached |
| `models/stair4_v2_1_heads.py` | Normalized phase encoder, real/complex GEMM parity, identity/learned key rotation, eta mixture, cosine control đúng số góc |
| `models/stair4_v2_1_utils.py` | Candidate dedup, train-positive mask, valid-row handling; tái sử dụng utility v2 sau test, không copy không cần thiết |
| `optimizers/stair4_v2_1_smoother.py` | Chỉ cần cho nhánh v2.1-S; composition H² sau BSC, parameter-free và step snapshot |
| `main_stair4_v2_1.py` | Parameter partition invariant, riêng loss diagnostics, seed/RNG, manifests và baseline evaluation semantics |
| `configs/dataset_stair4_v2_1_{baby,sports,electronics}.yaml` | Kế thừa giá trị dataset baseline; ghi explicit mọi override, primary xi=0, identity rotation |
| `tests/test_stair4_v2_1.py` | Algebra, gradient, baseline parity, ranking/checkpoint contracts |
| `notebook/P4/stair4_v2_1.ipynb` | Pinned checkout, subprocess preflight, smoke qua warmup+ramp, training/resume/exports; plots chỉ đọc metrics thật |
| `docs/giai_doan_4/STAIR4_v2_1_Experiment_Report.md` | Tạo khi có run thật; bảng còn trống trước khi đo, không điền forecast |

Không sửa `main.py` hoặc optimizer baseline để làm đối chứng khớp. Không kế thừa nguyên `encode()` v2 đang lỗi mà chỉ đổi tên class. Nếu dùng chung code mới thì phải có oracle độc lập từ baseline để tránh test lặp lại cùng bug.

### 11.1. Test bắt buộc trước full training

1. FSC oracle với asymmetric beta vector; λ(A)=1 invariant và endpoint L=0.
2. B1 vs B0 multi-step: embeddings, Adam moments, decay, scores và loss.
3. Encoder unit norm, finite gradient với input nhỏ/lớn; phát hiện zero-view theo contract.
4. Complex và real/imag logits + gradients tương đương; fidelity global-phase invariant, signed kernel không invariant.
5. Givens identity/norm preservation; omega nhận gradient nhiều bước sau warmup; case detach-after-G phải bị regression test bắt.
6. Candidate duplicates, nhiều positives, singleton/all-positive/empty valid rows; masks chỉ dùng train.
7. Dense/chunked loss và gradient tương đương; integration memory benchmark trên GPU tách khỏi unit tests.
8. Xi=0 exact BSC fast path; H² sparse-vs-dense oracle; spectral perturbation bound trên graph thỏa giả thiết.
9. Parameter groups disjoint + exhaustive; callback chỉ cho item, snapshot clear khi backward/step lỗi.
10. Checkpoint roundtrip gồm optimizer/RNG/schedule và continued-step parity; full/pool masking, NDCG selection đúng baseline.

## 12. Socratic stress test của bản v2.1 đã tinh chỉnh

| Câu hỏi khó | Trả lời có thể bảo vệ | Bằng chứng còn thiếu |
|---|---|---|
| Tăng điểm có phải chỉ sửa bug? | Có thể; B0/B1/F1 tách rõ phần sửa FSC | Multi-seed corrected baseline |
| Vì sao cần complex thay vì real? | Không cần về năng lực tính toán; đây là inductive bias của kernel | C0/C1/C2/C3 và real-backend equivalence |
| Stopgrad có làm view tự đuổi nhau? | Có; target vẫn đổi theo embeddings qua steps | Drift, collapse và ablation C5 |
| Positive edge đã có trong X1, có shortcut không? | Có thể; train-only không đồng nghĩa không shortcut | Nếu có gain: thêm same-view hoặc train-edge-dropout control ngân sách ngang |
| Mixture eta=0.25 có cơ sở tối ưu không? | Không; nó là pilot cố định để thử thành phần signed/fidelity | Validation sensitivity; không suy diễn optimum |
| Bounded update có đảm bảo NDCG tăng? | Không; bound chỉ kiểm soát mức thay đổi toán tử | S0/S1 và identity-mix |
| Tại sao giữ BSC khi muốn nâng cấp kiến trúc? | BSC là cơ chế baseline đã có; chưa có bằng chứng cần thay | Chỉ mở spectral khi baseline và core CL đã được tách |
| Nếu cosine thắng thì thesis thất bại? | Không; kết quả bác bỏ giả thuyết quantum là kết quả nghiên cứu hợp lệ | Báo cáo trung thực cả kết quả âm |

**Phán định methodology:** đề xuất QISF/HQCL nguyên bản cần sửa lớn trước triển khai vì sai equivalence baseline, norm-preservation, stop-gradient và validation design. v2.1 trong tài liệu này là phương án đủ cụ thể để triển khai và bác bỏ bằng thực nghiệm; chưa có bằng chứng cải thiện chỉ số. Kết luận nghiên cứu phải chờ các gate và ablation, không viết trước đoạn bảo vệ luận văn khẳng định thành công.

## 13. Tài liệu tham khảo và giới hạn xác minh

- **[R1]** Li, A., & Casiraghi, E. (2026). *Quantum-enhanced Representation Learning and Matching Learning for Recommendation*. WWW 2026, 5722–5730. [DOI](https://doi.org/10.1145/3774904.3792086). Metadata/abstract được xác nhận ở [Aalto](https://research.aalto.fi/en/publications/quantum-enhanced-representation-learning-and-matching-learning-fo/); không coi công thức trong đề xuất người dùng là trích từ toàn văn đã kiểm chứng.
- **[R2]** Debi, S., & Makmal, A. (2025). *Variational quantum recommendation system with embedded latent vectors*. Scientific Reports, 15, 32907. [Publisher](https://www.nature.com/articles/s41598-025-15869-x). Đối chiếu mục phương pháp và Figure 8; không tổng quát ablation nhỏ sang STAIR.
- **[R3]** Wang, W., Shi, J., Guan, N., Zhang, S., & Li, X. (2026). *QGHNN: A Quantum Graph Hamiltonian Neural Network*. IEEE TETCI, 10(1), 830–843. [DOI](https://doi.org/10.1109/TETCI.2025.3604791), [metadata tác giả](https://scholars.cityu.edu.hk/en/publications/qghnn-a-quantum-graph-hamiltonian-neural-network/). Online publication 2025; không trộn với author list preprint.
- **[R4]** Wang, W. (2025). *QGHNN: A quantum graph Hamiltonian neural network*. [arXiv:2501.07986](https://arxiv.org/abs/2501.07986). Abstract được kiểm tra; không phải một thí nghiệm độc lập xác nhận R3.

Các counterexample và bound trong §2, §4, §7 là suy luận toán học của audit này, không phải benchmark lấy từ các bài báo. Không có training mới, không có số gain v2.1, không có kiểm chứng superiority/SOTA trong lần viết tài liệu này.
