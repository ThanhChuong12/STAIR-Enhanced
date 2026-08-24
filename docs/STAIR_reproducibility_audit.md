# Audit Reproducibility: STAIR trên Kaggle

## 0. Nhận định quan trọng trước khi vào chi tiết

`stair.ipynb` **không tái implement** model, loss hay data pipeline của STAIR. Toàn bộ 10 cell chỉ làm việc setup môi trường, chuẩn bị data, và gọi `main.py` gốc của tác giả qua `subprocess`. Điều này thay đổi hoàn toàn phạm vi audit: không có code kiến trúc/loss nào của bạn để so khớp phương trình paper, vì bạn đang chạy đúng 100% code gốc. Rủi ro reproducibility ở đây nằm hoàn toàn ở lớp **orchestration**: cách gọi `main.py`, cách chuẩn bị data, version dependency, và cách thu thập log/kết quả.

Tôi đã clone repo gốc (`yhhe2004/STAIR`) và đối chiếu trực tiếp với `main.py`, các file config `.yaml`, và cấu trúc data zip thật để kiểm tra. Đồng thời tôi đã verify công thức FSC/BSC trong `main.py` và `optimizers/utils.py` khớp chính xác với Eq. (1)-(5) trong paper — nên phần model/loss bạn không cần chỉnh gì, chỉ cần gọi đúng.

**Tóm tắt mức độ hoàn thiện:**

| Hạng mục | % Alignment | Ghi chú |
|---|---|---|
| Model architecture / loss (FSC, BSC, whitening) | 100% | Dùng nguyên code gốc, đã verify khớp paper |
| Data pipeline (định dạng, tên thư mục) | ~50% | Cell 0 sai hoàn toàn; Cell 3 đúng nhưng chưa xóa cell sai |
| Cách gọi training (CLI args vs `--config`) | ~40% | Bỏ qua `--config yaml`, thiếu `monitors`/`which4best`, có flag không tồn tại |
| Environment/dependency pinning | ~50% | 2 lần cài đặt xung đột version freerec/torchdata |
| Evaluation protocol (best checkpoint theo NDCG@20) | Không xác nhận được | Phụ thuộc default của freerec khi thiếu yaml |
| Multi-seed / statistical significance (5 runs, p-value) | 0% | Notebook chỉ chạy 1 seed/dataset |
| Logging & error handling | Yếu | `subprocess.run` không kiểm tra return code, capture toàn bộ stdout vào RAM |

---

## 1. Discrepancy Breakdown

| Module | Paper Spec | Official Repo | My Notebook | Action Required |
|---|---|---|---|---|
| **Cách chạy training** | — | `python main.py --config configs/Amazon2014Baby_550_MMRec.yaml` | Cell 4: gọi `main.py` với các flag rời `--dataset --root --weight-decay --gamma --batch-size --epochs --eval-freq` | **Bỏ cách gọi CLI rời**, dùng `--config configs/<dataset>.yaml`. Flag `--eval-freq` **không tồn tại** trong `cfg` của `main.py` → freerec Parser (kế thừa argparse) sẽ raise lỗi unrecognized argument, khiến `subprocess.run` fail âm thầm vì bạn không kiểm tra `result.returncode` |
| **Data format** | Amazon Baby/Sports/Electronics, MMRec-preprocessed | Mỗi dataset là 1 thư mục tên chính xác `Amazon2014Baby_550_MMRec` chứa `train.txt`, `valid.txt`, `test.txt`, `textual_modality.pkl`, `visual_modality.pkl` (đã verify bằng cách unzip) | Cell 0: tạo `data/baby`, `data/sports`, `data/electronics` và copy file `.npy`/`.txt` bất kỳ tìm thấy — **sai hoàn toàn định dạng và tên thư mục**. Cell 3 (bản sửa sau): copy đúng vào `/kaggle/data/Amazon2014Baby_550_MMRec/...` — **đúng** | Xóa hẳn Cell 0 (dead code, gây nhầm lẫn, tốn thời gian clone/copy vô ích trên Kaggle). Giữ logic của Cell 3 |
| **root path** | `root: data` (yaml), tức thư mục `data/` nằm cạnh `main.py`, chứa các thư mục con tên dataset | như trên | Cell 4 truyền `--root /kaggle/data`, Cell 3 copy vào `/kaggle/data/...` — nhất quán, **đúng**, nhưng chỉ đúng nếu bạn không dùng `--config` (vì yaml có `root: data` khác giá trị này) | Nếu chuyển sang `--config yaml`, phải override `root` bằng `--root /kaggle/data` **sau** `--config` (freerec Parser cho phép override CLI đè lên yaml — cần verify thứ tự parse, xem mục 3) |
| **Hyperparameters (γ, weight_decay, batch_size)** | Table 4: Baby (γ=0.1, wd=0.3, bs=1024), Sports (γ=0.2, wd=0.1, bs=1024), Electronics (γ=0.4, wd=0.1, bs=4096) | khớp `configs/*.yaml` | Cell 4 truyền đúng 3 bộ giá trị này qua CLI — **đúng theo Table 4** | Không cần sửa số, nhưng nên dùng yaml làm nguồn chính thay vì hard-code lại trong Python để tránh sai sót khi copy tay |
| **Hyperparameters còn lại** (`embedding-dim=64`, `num-layers=3`, `lr=1e-3`, `optimizer=adamwsevo`, `num-neighbors='5-1'`) | cố định cho cả 3 dataset | default trong `cfg.set_defaults` của `main.py`, trùng với yaml | Không truyền, dùng default — **trùng nhau, đúng**, nhưng là trùng hợp chứ không phải chủ đích | Nên truyền tường minh hoặc dùng `--config` để tránh phụ thuộc default ngầm |
| **`monitors` / `which4best`** (chọn best checkpoint theo NDCG@20, log Recall@10/20, NDCG@10/20) | "báo cáo kết quả trên best checkpoint theo validation NDCG@20 qua 500 epochs" | định nghĩa trong yaml: `monitors: [LOSS, Recall@1, Recall@10, Recall@20, NDCG@10, NDCG@20]`, `which4best: NDCG@20` | Cell 4/Cell 8 **không dùng `--config`** nên các field này rơi về default nội bộ của freerec (không nằm trong `cfg.set_defaults` của `main.py`) — **không có gì đảm bảo là NDCG@20 được dùng để chọn checkpoint** | Đây là lỗ hổng nghiêm trọng nhất về evaluation protocol. Bắt buộc dùng `--config` hoặc tự thêm `--monitors ... --which4best NDCG@20` như flag CLI nếu freerec hỗ trợ |
| **Config JSON tự tạo (Cell 7)** | — | không tồn tại cơ chế đọc JSON, `main.py` chỉ nhận `--config <yaml>` | Cell 7 tạo `configs/{dataset}.json` với key tùy ý (`data_path`, `device`, `emb_size`, `topk`...) — **không hề được `main.py` đọc**, hoàn toàn không liên quan tới flow thật | Xóa Cell 7, hoặc nếu muốn giữ log config cho báo cáo thì đổi format sang yaml đúng schema của freerec và thực sự dùng `--config` |
| **Cell 8 (chạy lại `main.py --dataset baby/sports/electronics`)** | — | dataset phải là tên đầy đủ `Amazon2014Baby_550_MMRec` | Cell 8 truyền `--dataset baby` — `getattr(freerec.data.datasets, 'baby')` sẽ `AttributeError`, rơi vào nhánh `RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)` mà `cfg.tasktag` chưa chắc được set đúng cho MMRec — **rất dễ crash hoặc chạy sai dataset**, và không truyền γ/weight_decay/batch_size riêng cho từng dataset | Xóa hẳn Cell 8 — đây là bản nháp cũ bị bỏ sót, xung đột trực tiếp với Cell 4 (train lại từ đầu, tốn GPU-hour Kaggle vô ích) |
| **Dependency versions** | `env.sh` gốc: `freerec==0.8.5`, `torchdata=0.7.1`, PyG qua conda | như trên | Cell 0 cài `freerec==0.8.3` + `torchdata==0.6.1`; Cell 2 cài lại `freerec==0.9.5` + `torchdata==0.7.1` — **2 version freerec khác nhau, cả hai đều khác bản gốc 0.8.5** | Pin chính xác `freerec==0.8.5`. freerec 0.9.5 có thể đổi API nội bộ (`GenRecArch`, `criterions.BPRLoss`, `graph.get_knn_graph`, `dict_to_device`...) mà `main.py` gốc dựa vào — rủi ro silent behavior change, không chỉ lỗi crash |
| **`torch_geometric` qua pip** | cài qua conda (`pyg` channel), đã build sẵn cho đúng CUDA/torch | `pip install torch_geometric` (Cell 0/2) — không có `torch-scatter`/`torch-sparse` matching wheel | Có rủi ro ImportError hoặc lỗi CUDA mismatch trên Kaggle T4 | Dùng wheel index chính thức của PyG khớp version torch/cuda có sẵn trên Kaggle, xem mục 3 |
| **Seed / multi-run** | Table 2: "Paired t-test qua 5 independent runs" để tính p-value | `cfg.set_defaults(seed=1)`, freerec set seed toàn cục | Notebook chỉ chạy 1 lần/dataset, không lặp seed | Ghi rõ trong khóa luận đây là giới hạn (1 run do ràng buộc GPU-hour Kagg9-12h/session), không claim reproduce p-value. Nếu muốn gần đúng hơn: lặp 2-3 seed cho Baby/Sports (dataset nhỏ) là khả thi trong 12h, Electronics thì khó |
| **Logging/error handling** | — | — | `subprocess.run(..., capture_output=True)` giữ toàn bộ stdout/stderr trong RAM cho tới khi xong; `result` không được kiểm tra `returncode` hay in ra | Với Electronics (500 epochs, log dày) rủi ro tốn RAM không cần thiết trên Kaggle (giới hạn 13-30GB). Nên stream log ra file + kiểm tra returncode để phát hiện crash sớm thay vì "chạy xong" ba dataset mà không biết có dataset nào lỗi âm thầm |

---

## 2. Kaggle Reproduction Checklist

1. **Chỉ giữ 1 luồng setup duy nhất** — xóa Cell 0, Cell 8. Giữ Cell 2 (env) + Cell 3 (data) làm nền, sửa version freerec.
2. **Chuyển sang dùng `--config yaml`** thay vì liệt kê flag rời — đảm bảo `monitors`/`which4best` được set đúng như tác giả.
3. **Kiểm tra returncode + stream log ra file** thay vì `capture_output=True`.
4. **Pin đúng version**: `freerec==0.8.5`, `torchdata==0.7.1`, PyG cài qua wheel index đúng CUDA.
5. **Xóa Cell 7** (config JSON chết) hoặc chuyển nó thành nơi lưu **bản sao yaml thực sự dùng để train**, phục vụ mục đích ghi log cho khóa luận (traceability), không phải để `main.py` đọc.

## 3. Code Fix — Cell setup + training (thay thế Cell 0, 2, 4, 7, 8)

```python
import os, shutil, subprocess, sys

os.chdir('/kaggle/working')
if os.path.exists('STAIR'):
    shutil.rmtree('STAIR')
subprocess.run(['git', 'clone', '--depth', '1',
                 'https://github.com/yhhe2004/STAIR.git'], check=True)

# --- Pin đúng version gốc theo env.sh của tác giả ---
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                 'freerec==0.8.5', 'torchdata==0.7.1', 'nvidia-ml-py'], check=True)

# PyG: dùng wheel index khớp torch/cuda có sẵn trên Kaggle thay vì pip thường
import torch
TORCH_VER = torch.__version__.split('+')[0]
CUDA_TAG = 'cu' + torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                 'torch-geometric',
                 '-f', f'https://data.pyg.org/whl/torch-{TORCH_VER}+{CUDA_TAG}.html'], check=True)
```

```python
# --- Chuẩn bị data đúng cấu trúc gốc: data/<DATASET_NAME>/{train,valid,test}.txt, *_modality.pkl ---
DATA_ROOT = '/kaggle/working/STAIR/data'
os.makedirs(DATA_ROOT, exist_ok=True)

dataset_map = {
    'Amazon2014Baby_550_MMRec':        '/kaggle/input/datasets/rainyle/stair-datasets-mmrec/Amazon2014Baby_550_MMRec',
    'Amazon2014Sports_550_MMRec':      '/kaggle/input/datasets/rainyle/stair-datasets-mmrec/Amazon2014Sports_550_MMRec',
    'Amazon2014Electronics_550_MMRec': '/kaggle/input/datasets/rainyle/stair-datasets-mmrec/Amazon2014Electronics_550_MMRec',
}
for name, src in dataset_map.items():
    dest = os.path.join(DATA_ROOT, name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    required = {'train.txt', 'valid.txt', 'test.txt', 'textual_modality.pkl', 'visual_modality.pkl'}
    missing = required - set(os.listdir(dest))
    assert not missing, f"{name} thiếu file: {missing}"
    print(f"✅ {name} OK")
```

```python
# --- Chạy training đúng theo yaml gốc, override root cho Kaggle, log ra file, kiểm tra lỗi ---
import time

os.chdir('/kaggle/working/STAIR')
os.makedirs('/kaggle/working/logs', exist_ok=True)

runs = ['Amazon2014Baby_550_MMRec', 'Amazon2014Sports_550_MMRec', 'Amazon2014Electronics_550_MMRec']

for ds in runs:
    cfg_path = f'configs/{ds}.yaml'
    log_path = f'/kaggle/working/logs/{ds}.log'
    print(f"\n{'='*50}\n🚀 TRAINING: {ds}\n{'='*50}")

    t0 = time.time()
    with open(log_path, 'w') as logf:
        result = subprocess.run(
            ['python', 'main.py', '--config', cfg_path,
             '--root', DATA_ROOT],   # override root, các field khác (gamma, wd, bs, monitors, which4best...) lấy từ yaml
            stdout=logf, stderr=subprocess.STDOUT,
            cwd='/kaggle/working/STAIR'
        )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"❌ {ds} THẤT BẠI (returncode={result.returncode}). Xem log: {log_path}")
        # In 30 dòng cuối để debug ngay, không phải mở file riêng
        with open(log_path) as f:
            print(''.join(f.readlines()[-30:]))
    else:
        print(f"✅ {ds} hoàn tất sau {elapsed/60:.1f} phút. Log: {log_path}")
```

Ghi chú quan trọng: `--root` truyền sau `--config` để CLI override giá trị `root: data` trong yaml (freerec Parser theo chuẩn argparse ưu tiên giá trị CLI cuối cùng — nên test nhanh bằng `python main.py --config configs/Amazon2014Baby_550_MMRec.yaml --root /kaggle/working/STAIR/data --epochs 1` trước khi chạy full 500 epochs, để chắc override hoạt động và không có lỗi thiếu file trước khi tốn GPU-hour).

## 4. Ghi chú riêng cho phần khóa luận

Vì model/loss là code gốc không chỉnh sửa, phần "Reproducibility" trong `03_stair.tex` nên tập trung mô tả: (1) môi trường Kaggle T4, (2) cấu hình data/hyperparameter đúng theo yaml gốc, (3) giới hạn 1 run/dataset thay vì 5 runs như paper (do giới hạn GPU-hour), và (4) đối chiếu kết quả Recall@10/20, NDCG@10/20 thực đo được với Table 2 của paper — không cần đối chiếu phương trình vì đó là black-box của tác giả, chỉ cần đối chiếu **numerical fidelity** (chênh lệch % so với Table 2).
