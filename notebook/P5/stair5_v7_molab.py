import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🚀 GIAI ĐOẠN 5 — KIẾN TRÚC STAIR5-v7 / UCR-D: UPDATE-COMPATIBLE RETENTION WITH DOSE CONTROL
    ### 🏆 Kaggle / Colab / Molab ML Engineering Pipeline — Multimodal Recommendation with Update-Compatible Retention for Optimizer-Side Item Smoothing

    > **Đề tài:** Recommender Systems using Graph Representation: Multi-modal
    > **Khóa luận tốt nghiệp:** Khóa 2021–2025 — Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM
    > **Sinh viên thực hiện:** Lê Hà Thanh Chương (23120195) & Bùi Trung Hiếu (23120257)
    > **Giảng viên hướng dẫn:** TS. Nguyễn Ngọc Thảo
    > **Kiến trúc đề xuất:** **STAIR5-v7 / UCR-D** (*Update-Compatible Retention with Dose Control on Item BSC Semantic Branch*)
    > **Tài liệu đặc tả phương pháp luận:** `docs/giai_doan_5/STAIR5_v7_Report.md` (Authority Dated 2026-10-07)
    > **Mã nguồn lõi:** `models/stair5_v7.py`, `models/stair5_v7_graph.py`, `models/stair5_v7_utils.py`, `optimizers/stair5_v7_smoother.py`, `main_stair5_v7.py`
    > **Tập dữ liệu benchmark:** **Amazon Sports**, **Amazon Baby**, **Amazon Electronics** (3 tập chuẩn thương mại điện tử)
    > **Chiến lược thực thi:** **3-Tier Stage Execution** (`Pha A: Sports [Pilot] ➔ Pha B: Baby [Confirmatory] ➔ Pha C: Electronics [Scale-up]`)
    > **Hỗ trợ môi trường:** Hoàn toàn tương thích giữa **Kaggle Kernel**, **Google Colab**, và **Molab Workstation (Marimo reactive engine / NVIDIA RTX PRO 6000 Blackwell 96 GB)**

    ---

    ## 📑 1. BẢN THIẾT KẾ KIẾN TRÚC ĐỀ XUẤT: STAIR5-v7 / UCR-D

    | Thành Phần Trụ Cột | Cơ Chế Kỹ Thuật STAIR5-v7 (UCR-D) | Đột Phá & Ưu Thế So Với Baseline v4 & v6 |
    |:---|:---|:---|
    | **1. Kế Thừa Trọn Vẹn Backbone v4** | Modality Interaction SVD Whitening, Forward Stepwise Convolution ($L=3$), BPR Loss + Heterogeneous InfoNCE NLGCL objective. | Bảo toàn nguyên vẹn độ trễ suy luận $O(1)$; khôi phục bitwise v4 khi $\rho=0$ hoặc chọn arm `V4-control` / `V7-recovery`. |
    | **2. Bắt Hướng Cập Nhật Thực Tế $D_t$** | Trích xuất $D_t = \frac{m_t / (1 - \beta_1^t)}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}$ qua detached view từ item optimizer group. | Đánh giá trực tiếp xung đột tối ưu thực tế thay vì suy đoán gián tiếp từ gradient BPR hay EMA. |
    | **3. Ngưỡng Cường Độ Thích Ứng $\tau_{\mathrm{mag}}$** | $r_i = \|D_{t, i:}\|_2$, $\tau_{\mathrm{mag}} = \max\left(10^{-12}, \zeta \cdot \operatorname{median}_{r_i > 0}(r_i)\right)$ (mặc định $\zeta=0.01$). | Loại trừ nhiễu số học từ các hàng item có độ lớn cập nhật tiệm cận 0; chỉ can thiệp trên cạnh khả dụng. |
    | **4. Đo Lường Đối Kháng Từng Cặp $q_{ij}$** | $v_{ij}^{(t)} = \operatorname{clip}\left(\frac{D_{t, i:}^\top D_{t, j:}}{r_i r_j}, -1.0, 1.0\right)$, $q_{ij} = \max(0, -v_{ij})$ xử lý theo chunk ($C=4096$). | Bắt trọn xung đột hình học giữa các láng giềng semantic; chunk $C=4096$ giới hạn bộ nhớ scratchpad $<3.2\text{ MiB}$. |
    | **5. Hiệu Chuẩn Liều Lượng (Dose Control)** | Ramp tăng dần $\rho_t = \rho \cdot \min(1.0, \frac{e_t + p_t}{10.0})$, tải xung đột $Z_t$, suy giảm $\theta_t = \min(\theta_{\max}, \frac{\rho_t}{Z_t})$, $a_{ij} = \theta_t q_{ij}$. | Kiểm soát chính xác tổng lượng liên kết bị triệt giảm $\rho_{\mathrm{eff}} \le \rho_t$; cap $\theta_{\max}=0.25$ ngăn xóa cạnh cực đoan. |
    | **6. Hoàn Trả Khối Lượng Về Đường Chéo** | $h_{t, i} = \sum_{j \ne i} A_{t, ij}$, $W_R^{(t)} = W_0 - A_t + \operatorname{diag}(h_t) \implies W_R^{(t)} \mathbf{1} = d_0$. | Bảo toàn tuyệt đối bậc đỉnh hàng $d_0$; bù trừ lượng tương tác bị triệt giảm thành self-retention, co phổ $\|S_7\|_2 \le 1.0$. |
    | **7. Topology CSR Cố Định Trên GPU** | Duy trì một cấu trúc CSR thống nhất mở rộng (chứa $S_4$ + ô đường chéo); cập nhật in-place mảng `active_values`. | Loại bỏ hoàn toàn per-step COO coalescing, CPU synchronization, và rò rỉ GPU memory context. |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 1 ⚙️ Thiết lập Môi trường, Dependencies & Đồng bộ Mã Nguồn STAIR5-v7
    - Hỗ trợ kép cả môi trường **Molab Linux** (`/home/marimo`) và **Kaggle Kernel** (`/kaggle/working`).
    - Đồng bộ mã nguồn từ repository (bảo toàn mã nguồn hiện hữu, không chạy hard reset).
    - Cài đặt dependencies tương thích không sinh cảnh báo quyền cache: `freerec>=0.9.7`, `torch-geometric`, `nvidia-ml-py`, `prettytable`, `matplotlib`, `pyyaml`, `scipy`, `pandas`, `kagglehub`.
    - Kiểm tra GPU VRAM và chạy script xác thực môi trường trong tiến trình con.
    """)
    return


@app.cell
def _():
    # Cell 1: Môi trường, Dependencies & Xác thực Subprocess
    import os, shutil, subprocess, sys, glob, torch
    from pathlib import Path

    # 1. Đồng bộ thư mục làm việc (tương thích Molab / Kaggle / Colab)
    _safe_home = os.path.expanduser("~")
    _kw = '/kaggle/working'
    os.makedirs(_kw, exist_ok=True)

    # Tìm vị trí STAIR-Enhanced hiện hữu
    _possible_stair_dirs = [
        os.path.join(_safe_home, 'STAIR-Enhanced'),
        os.path.join(_kw, 'STAIR-Enhanced'),
        os.path.abspath('STAIR-Enhanced'),
        os.path.abspath('.'),
    ]
    _stair_dir = None
    for _p in _possible_stair_dirs:
        if os.path.isdir(_p) and os.path.exists(os.path.join(_p, 'models', 'stair5_v7.py')):
            _stair_dir = _p
            break

    if _stair_dir is None:
        _stair_dir = os.path.join(_kw, 'STAIR-Enhanced')
        if not os.path.exists(_stair_dir):
            print("📦 Đang clone repository STAIR-Enhanced (branch main)...")
            try:
                subprocess.run(
                    ['git', 'clone', '--depth', '1', 'https://github.com/ThanhChuong12/STAIR-Enhanced.git', _stair_dir],
                    check=True
                )
                print("✅ Clone thành công!")
            except Exception as _e:
                print(f"⚠️ Git clone gặp lỗi: {_e}")
        else:
            print("Thư mục STAIR-Enhanced đã tồn tại. Đang kiểm tra mã nguồn STAIR5-v7...")
            try:
                subprocess.run(['git', '-C', _stair_dir, 'pull', 'origin', 'main'], check=False, capture_output=True)
                print("✅ Đã kéo cập nhật mới nhất từ origin/main.")
            except Exception:
                pass

    # Tạo liên kết symlink 2 chiều giữa ~/STAIR-Enhanced và /kaggle/working/STAIR-Enhanced
    _home_stair = os.path.join(_safe_home, 'STAIR-Enhanced')
    _kw_stair = os.path.join(_kw, 'STAIR-Enhanced')
    if os.path.exists(_home_stair) and not os.path.exists(_kw_stair):
        try:
            os.symlink(_home_stair, _kw_stair)
        except Exception:
            pass
    elif os.path.exists(_kw_stair) and not os.path.exists(_home_stair):
        try:
            os.symlink(_kw_stair, _home_stair)
        except Exception:
            pass

    # Fallback từ Kaggle Input nếu chưa có models/stair5_v7.py
    if not os.path.exists(os.path.join(_stair_dir, 'models', 'stair5_v7.py')):
        _input_matches = glob.glob('/kaggle/input/**/models/stair5_v7.py', recursive=True)
        if _input_matches:
            _src_repo = os.path.dirname(os.path.dirname(_input_matches[0]))
            print(f"📦 Phát hiện mã nguồn STAIR5-v7 từ Kaggle Input: {_src_repo} -> Sao chép vào {_stair_dir}...")
            os.makedirs(_stair_dir, exist_ok=True)
            shutil.copytree(_src_repo, _stair_dir, dirs_exist_ok=True)

    _active_dir = _stair_dir if os.path.exists(_stair_dir) else os.path.abspath('.')
    os.chdir(_active_dir)

    for _p in [_active_dir, _kw_stair, _home_stair, _kw, '.']:
        if _p and os.path.exists(_p) and (_p not in sys.path):
            sys.path.insert(0, _p)

    _sub_env = os.environ.copy()
    _sub_env['PYTHONPATH'] = os.pathsep.join([_p for _p in [_active_dir, _kw_stair, _home_stair, os.environ.get('PYTHONPATH', '')] if _p])
    _sub_env['PYTHONWARNINGS'] = 'ignore'
    _sub_env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['PYTHONPATH'] = _sub_env['PYTHONPATH']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    # Xóa cache module v7 cũ trong kernel
    for _mod_name in list(sys.modules.keys()):
        if any(k in _mod_name for k in ['stair5_v7', 'models.stair5_v7', 'optimizers.stair5_v7_smoother']):
            sys.modules.pop(_mod_name, None)

    # 2. Cài đặt dependencies (bỏ qua torchdata vì Python 3.13 không hỗ trợ wheel, freerec_compat tự mock)
    print("Cài đặt dependencies tương thích (freerec, torch-geometric, pyyaml, prettytable, kagglehub)...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-q',
        'freerec>=0.9.7', 'torch-geometric', 'nvidia-ml-py', 'prettytable', 'matplotlib', 'pyyaml', 'scipy', 'pandas', 'kagglehub'
    ], check=False)

    # 3. Patch tương thích nội bộ PyTorch Dynamo / FreeRec
    try:
        import models.freerec_compat
    except Exception:
        try:
            import freerec_compat
        except Exception:
            pass

    # 4. Xác minh môi trường trong tiến trình con
    _verify_script = '''
    import torch, sys
    print("=== SUBPROCESS ENVIRONMENT VERIFICATION ===")
    print("Python Executable :", sys.executable)
    print("Python Version    :", sys.version.split()[0])
    print("PyTorch Version   :", torch.__version__)
    print("CUDA Available    :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Name       :", torch.cuda.get_device_name(0))
        print("Allocated VRAM    : {:.2f} MB".format(torch.cuda.memory_allocated() / (1024**2)))
    try:
        import freerec
        print("FreeRec Version   :", freerec.__version__)
    except Exception as e:
        print("FreeRec Import    : FAILED ({})".format(e))
    try:
        from models.stair5_v7 import STAIR5_v7_Model, ARMS_V7
        print("STAIR5-v7 Arms    :", list(ARMS_V7))
        print("STAIR5-v7 Core    : VERIFIED OK")
    except Exception as e:
        print("STAIR5-v7 Core    : FAILED ({})".format(e))
    print("===========================================")
    '''
    _res = subprocess.run([sys.executable, '-c', _verify_script], capture_output=True, text=True, env=_sub_env, cwd=_active_dir)
    print(_res.stdout)
    if _res.stderr and "warning" not in _res.stderr.lower():
        print("Stderr:", _res.stderr)

    print("✅ Khởi tạo môi trường STAIR5-v7 hoàn tất thành công.")
    return Path, os, shutil, subprocess, sys, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 2 📂 Chuẩn bị Dữ liệu (KaggleHub Auto-Fetch & Multi-Bridge sang /kaggle/data & Processed)
    - Tự động dò quét toàn bộ kho dữ liệu Kaggle Input (`/kaggle/input`), bộ nhớ cache hoặc thư mục hiện tại.
    - Nếu chưa có sẵn (ví dụ trên Molab), tự động nạp qua `kagglehub` slug `rainyle/stair-datasets-mmrec`.
    - Thiết lập liên kết symlink/copy đa hướng sang `/kaggle/data`, `/kaggle/data/Processed`, `STAIR-Enhanced/data` để FreeRec luôn tìm thấy dữ liệu.
    """)
    return


@app.cell
def _(os, shutil):
    # Cell 2: Chuẩn bị dữ liệu từ Kaggle Input / KaggleHub sang /kaggle/data & Processed
    _DATA_ROOT = '/kaggle/data'
    _PROCESSED_ROOT = os.path.join(_DATA_ROOT, 'Processed')
    _LOCAL_DATA = '/kaggle/working/STAIR-Enhanced/data'
    _LOCAL_PROCESSED = os.path.join(_LOCAL_DATA, 'Processed')
    _HOME_DATA = os.path.expanduser('~/STAIR-Enhanced/data')
    _HOME_PROCESSED = os.path.join(_HOME_DATA, 'Processed')
    for _d in [_DATA_ROOT, _PROCESSED_ROOT, _LOCAL_DATA, _LOCAL_PROCESSED, _HOME_DATA, _HOME_PROCESSED]:
        os.makedirs(_d, exist_ok=True)
    _TARGET_DATASETS = {'sports': ('Amazon2014Sports_550_MMRec', ['sport', 'sports', 'amazon2014sports']), 'baby': ('Amazon2014Baby_550_MMRec', ['baby', 'amazon2014baby']), 'electronics': ('Amazon2014Electronics_550_MMRec', ['electronic', 'electronics', 'amazon2014electronics'])}
    _REQUIRED_EXTENSIONS = ('.npy', '.pkl', '.txt', '.inter', '.item', '.pt', '.csv', '.yaml')

    def _bridge_directories(src_dir, target_folder):
        _destinations = [os.path.join(_DATA_ROOT, target_folder), os.path.join(_PROCESSED_ROOT, target_folder), os.path.join(_LOCAL_DATA, target_folder), os.path.join(_LOCAL_PROCESSED, target_folder), os.path.join(_HOME_DATA, target_folder), os.path.join(_HOME_PROCESSED, target_folder)]
        for _dst in _destinations:
            if os.path.abspath(src_dir) == os.path.abspath(_dst):
                continue
            os.makedirs(_dst, exist_ok=True)
            for _item in os.listdir(src_dir):
                _s_item = os.path.join(src_dir, _item)
                _d_item = os.path.join(_dst, _item)
                if os.path.isdir(_s_item):
                    if not os.path.exists(_d_item):
                        try:
                            os.symlink(_s_item, _d_item)
                        except Exception:
                            shutil.copytree(_s_item, _d_item, dirs_exist_ok=True)
                elif not os.path.exists(_d_item) or os.path.getsize(_d_item) == 0:
                    try:
                        os.symlink(_s_item, _d_item)
                    except Exception:
                        shutil.copy2(_s_item, _d_item)

    def _find_existing_dataset(folder_name, aliases):
        _search_bases = ['/kaggle/input', '/kaggle/data', '.', '..', '/kaggle/working', os.path.expanduser('~')]
        for _base in _search_bases:
            if not os.path.exists(_base):
                continue
            for _root, _dirs, _files in os.walk(_base):
                _bname = os.path.basename(_root).lower()
                if _bname == folder_name.lower() or any((_alias in _bname for _alias in aliases)):
                    if any((_f.endswith(_REQUIRED_EXTENSIONS) for _f in _files)):
                        return _root
        return None
    print('=' * 80)
    print('TIẾN TRÌNH DÒ QUÉT & LIÊN KẾT DỮ LIỆU TỰ ĐỘNG CHO STAIR5-v7 (UCR-D):')
    _missing = []
    for _key, (_folder, _aliases) in _TARGET_DATASETS.items():
        if not _find_existing_dataset(_folder, _aliases):
            _missing.append(_key)
    if _missing:
        print(f'📦 Phát hiện thiếu {_missing}. Đang kích hoạt KaggleHub auto-download...')
        try:
            import kagglehub
            _token = os.environ.get('KAGGLE_API_TOKEN', 'KGAT_4d9ab5bae4040ef4178b379d0262131b')
            _kaggle_cfg_dir = os.path.expanduser('~/.kaggle')
            os.makedirs(_kaggle_cfg_dir, exist_ok=True)
            with open(os.path.join(_kaggle_cfg_dir, 'access_token'), 'w') as _tf:
                _tf.write(_token)
            try:
                os.chmod(os.path.join(_kaggle_cfg_dir, 'access_token'), 384)
            except Exception:
                pass
            _slug = 'rainyle/stair-datasets-mmrec'
            print(f'  ↳ Đang tải dataset {_slug}...')
            _dl_path = kagglehub.dataset_download(_slug)
            print(f'  ↳ Tải thành công tại: {_dl_path}')
            os.makedirs('/kaggle/input', exist_ok=True)
            _in_symlink = '/kaggle/input/stair-datasets-mmrec'
            if not os.path.exists(_in_symlink) and (not os.path.islink(_in_symlink)):
                try:
                    os.symlink(_dl_path, _in_symlink)
                except Exception:
                    pass
            for _sub in os.listdir(_dl_path):
                _sub_full = os.path.join(_dl_path, _sub)
                _alias_in = os.path.join('/kaggle/input', _sub)
                if not os.path.exists(_alias_in) and (not os.path.islink(_alias_in)):
                    try:
                        os.symlink(_sub_full, _alias_in)
                    except Exception:
                        pass
        except Exception as _e:
            print(f'⚠️ KaggleHub auto-download cảnh báo: {_e}')
    _prepared_data = set()
    for _key, (_folder_name, _aliases) in _TARGET_DATASETS.items():
        _src = _find_existing_dataset(_folder_name, _aliases)
        if _src:
            _bridge_directories(_src, _folder_name)
            _prepared_data.add(_key)
            _cnt = len(os.listdir(_src))
            print(f'  ✅ [SẴN SÀNG] {_key.upper():12s} -> Nguồn: {_src} ({_cnt} files)')
        else:
            print(f'  ⚠️ [CHƯA THẤY] {_key.upper():12s} -> Sẽ kiểm tra lại thư mục nguồn.')
    print('=' * 80)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 3 🧪 Bộ Kiểm Thử Độc Lập Toàn Diện Cho STAIR5-v7 UCR-D (17 Unit Tests)
    - Thực thi nghiệm thu toán học và pipeline tích hợp trước khi khởi động huấn luyện.
    - Kiểm tra tính bảo toàn bậc đỉnh hàng $\sum_j W_{R, ij} = d_{0, i}$, tính co phổ $\|S_7\|_2 \le 1.0$, tính triệt tiêu của phép co giãn hằng số, khôi phục bitwise v4 khi $\rho=0$, sự cô lập của optimizer parameter groups, bộ nhớ gate và vòng đời snapshot GPU.
    - **Fail-fast gate:** Nếu bất kỳ unit test nào thất bại, dừng ngay lập tức.
    """)
    return


@app.cell
def _(os, subprocess, sys):
    # Cell 3: Chạy Bộ Kiểm Thử Độc Lập Toàn Diện Cho STAIR5-v7 (17 Unit Tests)
    print('=' * 80)
    print('🚀 CHẠY BỘ KIỂM THỬ ĐỘC LẬP TOÀN DIỆN CHO STAIR5-v7 UCR-D (17 UNIT TESTS)...')
    print('=' * 80)
    _test_cmd = [sys.executable, '-m', 'pytest', 'tests/test_stair5_v7_graph.py', 'tests/test_stair5_v7_pipeline.py', '-v']
    _sub_env = os.environ.copy()
    _sub_env['PYTHONPATH'] = os.pathsep.join([_p for _p in ['.', os.environ.get('PYTHONPATH', '')] if _p])
    _test_cwd = '/kaggle/working/STAIR-Enhanced' if os.path.exists('/kaggle/working/STAIR-Enhanced') else '.'
    _res = subprocess.run(_test_cmd, capture_output=True, text=True, env=_sub_env, cwd=_test_cwd)
    print(_res.stdout)
    if _res.returncode == 0:
        print('🎉 TẤT CẢ 17/17 UNIT TESTS CỦA STAIR5-v7 ĐÃ VƯỢT QUA XUẤT SẮC! SẴN SÀNG HUẤN LUYỆN.')
    else:
        if _res.stderr:
            print('STDERR:\n', _res.stderr)
        raise RuntimeError(f'Kiểm thử tiền trạm thất bại với mã lỗi {_res.returncode}; dừng huấn luyện để bảo vệ tài nguyên.')
    print('=' * 80)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 4 🛠️ Telemetry Engine: Training Runner, GPU Hardware Profiler & Visualizer
    - Quản lý tiến trình huấn luyện `main_stair5_v7.py` qua tiến trình con độc lập (`subprocess.Popen`), stream stdout thời gian thực.
    - Giám sát VRAM phần cứng (`nvidia-ml-py` / NVML) và trích xuất `training_telemetry.jsonl`.
    - Trích xuất tự động `selected_test_metrics.json` từ checkpoint được chọn theo validation NDCG@20.
    """)
    return


@app.cell
def _(Path, STAIR5_V7_CONFIGS, os, subprocess, sys, torch):
    # Cell 4: Telemetry Engine — Training Runner, Hardware Profiler & Visualization
    import time, re, json, threading
    import numpy as np
    import matplotlib.pyplot as plt
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    TRACKED_METRICS = ['Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20']
    BASELINE_REF = {'sports': {'Recall@10': 0.0743, 'Recall@20': 0.1111, 'NDCG@10': 0.0405, 'NDCG@20': 0.05}, 'baby': {'Recall@10': 0.0674, 'Recall@20': 0.1042, 'NDCG@10': 0.0359, 'NDCG@20': 0.0454}, 'electronics': {'Recall@10': 0.0442, 'Recall@20': 0.0665, 'NDCG@10': 0.0246, 'NDCG@20': 0.0303}}
    V4_REF = {'sports': {'Recall@10': 0.0765, 'Recall@20': 0.1143, 'NDCG@10': 0.0419, 'NDCG@20': 0.0517}, 'baby': {'Recall@10': 0.0678, 'Recall@20': 0.1056, 'NDCG@10': 0.0364, 'NDCG@20': 0.0461}, 'electronics': {'Recall@10': 0.0458, 'Recall@20': 0.0678, 'NDCG@10': 0.026, 'NDCG@20': 0.0316}}
    V6_REF = {'sports': {'Recall@10': 0.0765, 'Recall@20': 0.1143, 'NDCG@10': 0.0419, 'NDCG@20': 0.0517, 'epoch': 485}, 'baby': {'Recall@10': 0.0677, 'Recall@20': 0.1056, 'NDCG@10': 0.0364, 'NDCG@20': 0.0461, 'epoch': 480}, 'electronics': {'Recall@10': 0.0458, 'Recall@20': 0.0681, 'NDCG@10': 0.0259, 'NDCG@20': 0.0317, 'epoch': 440}}
    DATASET_PROFILES = {'sports': {'name': 'Amazon Sports', 'domain': 'E-commerce (Visual + Textual)', 'scale': '35,598 Users | 18,357 Items | 296K Interactions', 'color': '#ff7f0e', 'approx_mins': 42.0}, 'baby': {'name': 'Amazon Baby', 'domain': 'E-commerce (Visual + Textual)', 'scale': '19,445 Users | 7,050 Items | 160K Interactions', 'color': '#1f77b4', 'approx_mins': 20.0}, 'electronics': {'name': 'Amazon Electronics', 'domain': 'E-commerce (Visual + Textual)', 'scale': '192,403 Users | 63,001 Items | 1.69M Interactions', 'color': '#2ca02c', 'approx_mins': 345.0}}
    # Bảng đối chuẩn đa thế hệ lịch sử trên cả 3 tập dữ liệu
    vram_profile = {}
    STAIR5_V7_RESULTS = {}

    def vram_monitor(key, stop_evt, interval=2.0):
        """Background thread tracking host GPU allocated memory via NVML."""
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            records = []
            while not stop_evt.is_set():
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                records.append(mem.used / 1024 ** 2)
                time.sleep(interval)
            pynvml.nvmlShutdown()
            vram_profile[key] = records
        except Exception:
            vram_profile[key] = []

    def resolve_path(p):
        if os.path.exists(p):
            return p
        base = os.path.basename(p)
        candidates = [p, os.path.join('/kaggle/working/logs/stair5_v7', base), os.path.join('/kaggle/working/logs/GD5', base), os.path.join('logs/stair5_v7', base), os.path.expanduser(f'~/logs/stair5_v7/{base}')]
        for c in candidates:
            if os.path.exists(c):
                return c
        return p

    def parse_telemetry_jsonl(artifact_dir):
        p = Path(artifact_dir) / 'training_telemetry.jsonl'
        if not p.is_file():
            return []
        records, seen = ([], set())
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ep = data.get('epoch')
                    if ep not in seen:
                        seen.add(ep)
                        records.append(data)
                except Exception:
                    continue
        return records

    def extract_best_validation(log_path):
        log_path = resolve_path(log_path)
        if not os.path.exists(log_path):
            return (None, {})
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        matches = re.findall('VALID\\s+@Epoch:\\s*(\\d+).*?NDCG@20\\s+Avg:\\s*([0-9.]+)', content, re.DOTALL)
        if not matches:
            return (None, {})
        best_ep, best_val = (-1, -1.0)
        for ep_str, val_str in matches:
            ep, val = (int(ep_str), float(val_str))
            if val > best_val:
                best_val = val
                best_ep = ep
        return (best_ep, {'NDCG@20': best_val})

    def extract_test_metrics(log_path, artifact_dir=None):
        if artifact_dir:
            m_file = Path(artifact_dir) / 'selected_test_metrics.json'
            if m_file.is_file():
                try:
                    payload = json.loads(m_file.read_text(encoding='utf-8'))
                    raw_m = payload.get('metrics', {})
                    norm_m = {}
                    for k, v in raw_m.items():
                        norm_k = k
                        for tm in TRACKED_METRICS:
                            if k.lower() == tm.lower():
                                norm_k = tm
                                break
                        norm_m[norm_k] = v
                    return (payload.get('epoch'), norm_m)
                except Exception:
                    pass
        log_path = resolve_path(log_path)
        if not os.path.exists(log_path):
            return (None, {})
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        best_matches = list(re.finditer('\\[Coach\\]\\s*>>>\\s*Load best model\\s*@Epoch:\\s*(\\d+)', content))
        best_match = best_matches[-1] if best_matches else None
        if not best_match:
            return (None, {})
        content_after = content[best_match.end():]
        test_match = re.search('TEST\\s+@Epoch:\\s*(\\d+)(.*?)(?:=================|$)', content_after, re.DOTALL)
        if not test_match:
            return (None, {})
        test_epoch = int(test_match.group(1))
        test_snippet = test_match.group(2)
        test_metrics = {}
        for metric in TRACKED_METRICS:
            m = re.search(f'{metric}(?:\\s*Avg)?\\s*[:\\s]+\\s*([0-9.]+)', test_snippet, re.IGNORECASE)
            if m:
                test_metrics[metric] = float(m.group(1))
        return (test_epoch, test_metrics)

    def parse_training_losses(log_path, telemetry=None):
        if telemetry and len(telemetry) > 0:
            bpr_losses = [(r['epoch'], float(r['bpr_loss'])) for r in telemetry if 'bpr_loss' in r]
            cl_losses = [(r['epoch'], float(r['cl_loss'])) for r in telemetry if 'cl_loss' in r]
            total_losses = [(r['epoch'], float(r['total_loss'])) for r in telemetry if 'total_loss' in r]
            return (bpr_losses, cl_losses, total_losses)
        log_path = resolve_path(log_path)
        bpr_losses, cl_losses, total_losses = ([], [], [])
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            matches = re.findall('LOSS\\s+Avg:\\s*([0-9.]+)', content)
            for i, val in enumerate(matches):
                total_losses.append((i + 1, float(val)))
        return (bpr_losses, cl_losses, total_losses)

    def parse_valid_metric(log_path, metric='NDCG@20'):
        log_path = resolve_path(log_path)
        if not os.path.exists(log_path):
            return []
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        pattern = f'VALID\\s+@Epoch:\\s*(\\d+).*?{metric}\\s+Avg:\\s*([0-9.]+)'
        matches = re.findall(pattern, content, re.IGNORECASE)
        return [(int(ep), float(v)) for ep, v in matches]

    def run_training_stair5_v7(key, config_yaml=None, data_root=None, log_path=None, artifact_dir=None, **kwargs):
        """Executes STAIR5-v7 UCR-D training with real-time streaming and fail-fast handling."""
        cfg_dict = STAIR5_V7_CONFIGS.get(key, {}).copy()
        cfg_dict.update(kwargs)
        config_yaml = config_yaml or cfg_dict.get('yaml')
        label = f"{key}_{cfg_dict.get('v7_arm', 'UCR-D')}_seed{cfg_dict.get('seed', 1)}"
        log_path = log_path or str(Path(cfg_dict['log']).parent / (label + '.log'))
        artifact_dir = artifact_dir or str(Path(cfg_dict['artifact_dir']).parent / label)
        resuming = bool(cfg_dict.get('resume_from'))
        if not resuming and (Path(log_path).exists() or Path(artifact_dir).exists()):
            print(f'⚠️ Thông báo: Ghi đè hoặc tái lập kịch bản run cho {label}.')
            if Path(artifact_dir).exists():
                for stale in ('manifest.json', 'training_checkpoint.pt', 'training_telemetry.jsonl', 'best_model.pt', 'last_model.pt', 'selected_test_metrics.json'):
                    sp_file = Path(artifact_dir) / stale
                    if sp_file.exists():
                        try:
                            sp_file.unlink()
                        except OSError:
                            pass
        STAIR5_V7_CONFIGS[key].update(log=log_path, artifact_dir=artifact_dir, v7_arm=cfg_dict.get('v7_arm', 'UCR-D'))
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(artifact_dir, exist_ok=True)
        print('=' * 80)
        print(f'🚀 KHỞI ĐỘNG TIẾN TRÌNH HUẤN LUYỆN STAIR5-v7 / UCR-D: {key.upper()}')
        print(f'  * Dataset Key         : {key}')
        print(f'  * YAML Configuration  : {config_yaml}')
        print(f'  * Log Path            : {log_path}')
        print(f'  * Artifact Directory  : {artifact_dir}')
        print(f"  * V7 Arm              : {cfg_dict.get('v7_arm', 'UCR-D')}")
        print(f"  * Target Dose (ρ)     : {cfg_dict.get('v7_rho', 0.01)}")
        print(f"  * Attenuation Cap (θ) : {cfg_dict.get('v7_theta_max', 0.25)}")
        print(f"  * Magnitude Mult (ζ)  : {cfg_dict.get('v7_zeta', 0.01)}")
        print(f"  * Operator Blend (η)  : {cfg_dict.get('eta', 0.1)}")
        print(f"  * Seed                : {cfg_dict.get('seed', 1)}")
        print('=' * 80)
        stop_evt = threading.Event()
        th = threading.Thread(target=vram_monitor, args=(key, stop_evt), daemon=True)
        th.start()
        t0 = time.time()
        _possible_runners = ['/kaggle/working/STAIR-Enhanced/main_stair5_v7.py', os.path.expanduser('~/STAIR-Enhanced/main_stair5_v7.py'), 'main_stair5_v7.py']
        runner_py = next((r for r in _possible_runners if os.path.exists(r)), 'main_stair5_v7.py')
        if not os.path.exists(config_yaml):
            cand_y1 = os.path.join('configs', os.path.basename(config_yaml))
            cand_y2 = os.path.expanduser(f'~/STAIR-Enhanced/configs/{os.path.basename(config_yaml)}')
            cand_y3 = f'/kaggle/working/STAIR-Enhanced/configs/{os.path.basename(config_yaml)}'
            for cand in [cand_y1, cand_y2, cand_y3]:
                if os.path.exists(cand):
                    config_yaml = cand
                    break
        if data_root is None:
            data_root = '/kaggle/data' if os.path.exists('/kaggle/data') else 'data'
        device_str = str(cfg_dict.get('device', '0' if torch.cuda.is_available() else 'cpu'))
        cmd = [sys.executable, '-u', str(runner_py), '--config', str(config_yaml), '--root', str(data_root), '--device', str(device_str), '--ranking', 'full', '--artifact-dir', str(artifact_dir)]
        defaults = {'v7_arm': 'UCR-D', 'v7_rho': 0.01, 'v7_theta_max': 0.25, 'v7_zeta': 0.01, 'v7_pair_chunk_size': 4096, 'v7_warmup_epochs': 10.0, 'v7_shuffled_seed': 1, 'seed': 1, 'eval_freq': 5, 'eta': 0.1, 'k_cf': 5, 'c_min': 2, 't_shrinkage': 5.0, 'cf_block_size': 64, 'cf_memory_budget_mib': 128, 'lambda_nlgcl': 0.01, 'nlgcl_tau': 0.2, 'nlgcl_G': 1, 'nlgcl_alpha': 0.5, 'cl_chunk_size': 1024, 'knn_chunk_size': 1024, 'knn_device': 'auto', 'graph_cache_dir': '/kaggle/working/runs/stair5_v7/graph_cache'}
        settings = {**defaults, **cfg_dict}
        for k_item in defaults:
            cmd += ['--' + k_item.replace('_', '-'), str(settings[k_item])]
        for k_item in ('epochs', 'batch_size', 'gamma', 'lr', 'weight_decay', 'num_workers'):
            if k_item in cfg_dict:
                cmd += ['--' + k_item.replace('_', '-'), str(cfg_dict[k_item])]
        if cfg_dict.get('resume_from'):
            cmd += ['--resume-from', str(cfg_dict['resume_from'])]
        print('Command:', ' '.join(cmd))
        env = os.environ.copy()
        _active_dir = os.path.dirname(os.path.abspath(runner_py))
        env['PYTHONPATH'] = os.pathsep.join([p for p in ['.', _active_dir, os.environ.get('PYTHONPATH', '')] if p])
        env['PYTHONUTF8'] = '1'
        try:
            with open(log_path, 'a' if resuming else 'w', encoding='utf-8') as lf:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1, env=env)
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    lf.write(line)
                    lf.flush()
                proc.wait()
                ret_code = proc.returncode
        finally:
            stop_evt.set()
            th.join(timeout=3.0)
        elapsed = time.time() - t0
        print(f'⏱️ Thời gian thực thi ({key.upper()}): {elapsed / 60:.2f} phút | Trạng thái exit code: {ret_code}')
        if ret_code != 0:
            raise RuntimeError(f'Tiến trình huấn luyện {key.upper()} thất bại với mã lỗi {ret_code}. Kiểm tra log: {log_path}')
        return elapsed  # Tìm file runner main_stair5_v7.py  # Tìm file config yaml  # Xác định data_root phù hợp

    return (
        BASELINE_REF,
        STAIR5_V7_RESULTS,
        TRACKED_METRICS,
        V4_REF,
        V6_REF,
        extract_best_validation,
        extract_test_metrics,
        json,
        parse_telemetry_jsonl,
        parse_training_losses,
        parse_valid_metric,
        plt,
        run_training_stair5_v7,
        time,
        vram_profile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 5 📋 Cấu hình Siêu tham số STAIR5-v7 UCR-D (Dataset-Adaptive Matrix & Blackwell Tuning)
    - Ma trận cấu hình chi tiết cho 3 tập benchmark theo đặc tả toán học trong `docs/giai_doan_5/STAIR5_v7_Report.md`:
      * **Amazon Sports**: $\rho = 0.01, \theta_{\max} = 0.25, \zeta = 0.01, \gamma = 0.2, wd = 0.1, \text{lr} = 0.001, \text{batch\_size} = 1024$.
      * **Amazon Baby**: $\rho = 0.01, \theta_{\max} = 0.25, \zeta = 0.01, \gamma = 0.1, wd = 0.3, \text{lr} = 0.001, \text{batch\_size} = 1024$.
      * **Amazon Electronics**: $\rho = 0.01, \theta_{\max} = 0.25, \zeta = 0.01, \gamma = 0.4, wd = 0.1, \text{lr} = 0.001, \text{batch\_size} = 4096$.
      * **Blackwell Hardware Scaling**: Tự động nhận diện GPU dung lượng lớn (NVIDIA RTX PRO 6000 Blackwell 96 GB GDDR7) để mở rộng chunk size (`cl_chunk_size = 1024`, `pair_chunk_size = 8192`), loại bỏ thắt cổ chai vòng lặp Python!
    """)
    return


@app.cell
def _(os, torch):
    # Cell 5: Cấu hình Siêu tham số STAIR5-v7 UCR-D (Dataset-Adaptive Matrix)
    import uuid
    from datetime import datetime
    RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    for _d in ['/kaggle/working/logs/stair5_v7', '/kaggle/working/reports', '/kaggle/working/runs/stair5_v7/graph_cache', os.path.expanduser('~/logs/stair5_v7'), os.path.expanduser('~/reports')]:
        try:
            os.makedirs(_d, exist_ok=True)
        except Exception:
            pass
    _is_blackwell = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 32 * 1024 ** 3
    _elec_cl_chunk = 1024 if _is_blackwell else 128
    _elec_pair_chunk = 8192 if _is_blackwell else 1024
    # Nhận diện GPU VRAM: Nếu > 32GB VRAM (Blackwell RTX PRO 6000 96GB), tự động mở rộng chunk size để đạt hiệu năng tối đa
    if _is_blackwell:
        print(f'🚀 PHÁT HIỆN GPU DUNG LƯỢNG LỚN: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB VRAM).')
        print(f'  ↳ Tối ưu Electronics: cl_chunk_size={_elec_cl_chunk}, pair_chunk_size={_elec_pair_chunk} (Tăng tốc tối đa).')
    STAIR5_V7_CONFIGS = {'sports': {'yaml': '/kaggle/working/STAIR-Enhanced/configs/Amazon2014Sports_STAIR5_v7.yaml', 'log': f'/kaggle/working/logs/stair5_v7/{RUN_ID}/sports_stair5_v7.log', 'artifact_dir': f'/kaggle/working/runs/stair5_v7/{RUN_ID}/sports_UCR-D_seed1', 'v7_arm': 'UCR-D', 'v7_rho': 0.01, 'v7_theta_max': 0.25, 'v7_zeta': 0.01, 'v7_pair_chunk_size': 4096, 'v7_warmup_epochs': 10.0, 'eta': 0.1, 'k_cf': 5, 'c_min': 2, 't_shrinkage': 5.0, 'cf_block_size': 64, 'lambda_nlgcl': 0.01, 'nlgcl_tau': 0.2, 'nlgcl_G': 1, 'nlgcl_alpha': 0.5, 'cl_chunk_size': 1024, 'seed': 1, 'eval_freq': 5}, 'baby': {'yaml': '/kaggle/working/STAIR-Enhanced/configs/Amazon2014Baby_STAIR5_v7.yaml', 'log': f'/kaggle/working/logs/stair5_v7/{RUN_ID}/baby_stair5_v7.log', 'artifact_dir': f'/kaggle/working/runs/stair5_v7/{RUN_ID}/baby_UCR-D_seed1', 'v7_arm': 'UCR-D', 'v7_rho': 0.01, 'v7_theta_max': 0.25, 'v7_zeta': 0.01, 'v7_pair_chunk_size': 4096, 'v7_warmup_epochs': 10.0, 'eta': 0.1, 'k_cf': 5, 'c_min': 2, 't_shrinkage': 5.0, 'cf_block_size': 64, 'lambda_nlgcl': 0.01, 'nlgcl_tau': 0.2, 'nlgcl_G': 1, 'nlgcl_alpha': 0.5, 'cl_chunk_size': 1024, 'seed': 1, 'eval_freq': 5}, 'electronics': {'yaml': '/kaggle/working/STAIR-Enhanced/configs/Amazon2014Electronics_STAIR5_v7.yaml', 'log': f'/kaggle/working/logs/stair5_v7/{RUN_ID}/electronics_stair5_v7.log', 'artifact_dir': f'/kaggle/working/runs/stair5_v7/{RUN_ID}/electronics_UCR-D_seed1', 'v7_arm': 'UCR-D', 'v7_rho': 0.01, 'v7_theta_max': 0.25, 'v7_zeta': 0.01, 'v7_pair_chunk_size': _elec_pair_chunk, 'v7_warmup_epochs': 10.0, 'eta': 0.1, 'k_cf': 5, 'c_min': 2, 't_shrinkage': 5.0, 'cf_block_size': 64, 'lambda_nlgcl': 0.01, 'nlgcl_tau': 0.2, 'nlgcl_G': 1, 'nlgcl_alpha': 0.5, 'cl_chunk_size': _elec_cl_chunk, 'seed': 1, 'eval_freq': 5}}
    print('✅ Ma trận cấu hình STAIR5-v7 UCR-D đã nạp thành công:')
    for _k, _v in STAIR5_V7_CONFIGS.items():
        print(f"  * {_k.upper():12s} -> Arm: {_v['v7_arm']} | Config: {os.path.basename(_v['yaml'])} | Rho: {_v['v7_rho']} | ThetaMax: {_v['v7_theta_max']}")
    return (STAIR5_V7_CONFIGS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 6a 🏋️ Huấn luyện Pha A — Amazon Sports (Pilot STAIR5-v7 / UCR-D)
    - Huấn luyện mô hình STAIR5-v7 trên tập **Amazon Sports** (35,598 Users, 18,357 Items).
    - Ghi nhận telemetry từng epoch: BPR loss, NLGCL InfoNCE loss, VRAM phân bổ, và telemetry xung đột $Z_t, \theta_t, \rho_{\mathrm{eff}}$.
    """)
    return


@app.cell
def _(
    BASELINE_REF,
    STAIR5_V7_CONFIGS,
    STAIR5_V7_RESULTS,
    TRACKED_METRICS,
    V4_REF,
    V6_REF,
    extract_best_validation,
    extract_test_metrics,
    run_training_stair5_v7,
):
    # Cell 6a: Training STAIR5-v7 UCR-D on Amazon Sports (Pha A: Pilot)
    sports_time = run_training_stair5_v7('sports')
    ep_sports, metrics_sports = extract_test_metrics(STAIR5_V7_CONFIGS['sports']['log'], STAIR5_V7_CONFIGS['sports']['artifact_dir'])
    val_ep_sports, val_m_sports = extract_best_validation(STAIR5_V7_CONFIGS['sports']['log'])
    STAIR5_V7_RESULTS['sports'] = {
        'metrics': metrics_sports,
        'best_epoch': ep_sports,
        'val_metrics': val_m_sports,
        'train_time': sports_time,
    }

    print('\n' + '=' * 80)
    print('📊 KẾT QUẢ THỰC NGHIỆM PHA A — AMAZON SPORTS (STAIR5-v7 / UCR-D):')
    print(f"  * Checkpoint tối ưu tại Epoch: {ep_sports} (Validation NDCG@20 = {val_m_sports.get('NDCG@20', 0.0):.4f})")
    for _m in TRACKED_METRICS:
        _val = metrics_sports.get(_m, 0.0)
        _base = BASELINE_REF['sports'].get(_m, 0.0)
        _v4_val = V4_REF['sports'].get(_m, 0.0)
        _v6_val = V6_REF['sports'].get(_m, 0.0)
        _d_base = ((_val - _base) / _base) * 100 if _base > 0 else 0.0
        _d_v4 = ((_val - _v4_val) / _v4_val) * 100 if _v4_val > 0 else 0.0
        _d_v6 = ((_val - _v6_val) / _v6_val) * 100 if _v6_val > 0 else 0.0
        print(f"  * {_m:10s} : {_val:.4f}  (vs Baseline: {_d_base:+.2f}% | vs v4: {_d_v4:+.2f}% | vs v6: {_d_v6:+.2f}%)")
    print('=' * 80)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 6b ⚡ Biểu đồ Tiêu thụ Bộ nhớ Tensor Mô hình — Amazon Sports
    - Phân tích chi tiết mức tiêu thụ VRAM thực tế (`max_memory_allocated_bytes`) trên Amazon Sports.
    """)
    return


@app.cell
def _(STAIR5_V7_CONFIGS, os, parse_telemetry_jsonl, plt, vram_profile):
    # Cell 6b: Model Tensor VRAM Profile — Amazon Sports (Paper Standard)
    _sports_art = STAIR5_V7_CONFIGS['sports']['artifact_dir']
    _telemetry_sports = parse_telemetry_jsonl(_sports_art)

    plt.figure(figsize=(9, 4.5), dpi=150)
    if _telemetry_sports:
        _eps = [r['epoch'] for r in _telemetry_sports]
        _vram_mb = [r.get('max_memory_allocated_bytes', 0) / (1024**2) for r in _telemetry_sports]
        plt.plot(_eps, _vram_mb, color='#ff7f0e', lw=2.2, marker='o', ms=4, label='Peak Allocated VRAM (MB)')
        plt.title('Amazon Sports — Peak Model Tensor VRAM Profile (STAIR5-v7)', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Allocated VRAM (MB)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=True)
    elif 'sports' in vram_profile and len(vram_profile['sports']) > 0:
        plt.plot(vram_profile['sports'], color='#ff7f0e', lw=1.8, label='Host NVML GPU Used (MB)')
        plt.title('Amazon Sports — Host GPU Memory Profile (NVML)', fontsize=12, fontweight='bold')
        plt.xlabel('Polling Step (~2s)', fontsize=10)
        plt.ylabel('Memory (MB)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=True)
    else:
        plt.text(0.5, 0.5, 'Chưa có dữ liệu VRAM hoặc NVML không khả dụng', ha='center', va='center')

    plt.tight_layout()
    os.makedirs('/kaggle/working/reports', exist_ok=True)
    plt.savefig('/kaggle/working/reports/vram_profile_sports.png')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 6c 📈 Động Lực Học & Quá Trình Hội Tụ — Amazon Sports
    - Trực quan hóa hàm mất mát (Total loss, BPR loss) và đường cong Validation NDCG@20 so với v4 và v6.
    """)
    return


@app.cell
def _(
    BASELINE_REF,
    STAIR5_V7_CONFIGS,
    V4_REF,
    V6_REF,
    parse_telemetry_jsonl,
    parse_training_losses,
    parse_valid_metric,
    plt,
):
    # Cell 6c: Learning Dynamics & Convergence Profiles — Amazon Sports
    _fig, _ax1 = plt.subplots(figsize=(10, 4.8), dpi=150)

    _sports_art = STAIR5_V7_CONFIGS['sports']['artifact_dir']
    _telemetry_sports = parse_telemetry_jsonl(_sports_art)
    _log_sports = STAIR5_V7_CONFIGS['sports']['log']
    _bpr_l, _cl_l, _tot_l = parse_training_losses(_log_sports, _telemetry_sports)
    _val_ndcg = parse_valid_metric(_log_sports, 'NDCG@20')

    if _tot_l:
        _x_tot, _y_tot = zip(*_tot_l)
        _ax1.plot(_x_tot, _y_tot, color='#1f77b4', lw=2.0, label='Total Loss (BPR + λ·NLGCL)')
    if _bpr_l:
        _x_bpr, _y_bpr = zip(*_bpr_l)
        _ax1.plot(_x_bpr, _y_bpr, color='#1f77b4', lw=1.5, ls='--', alpha=0.7, label='BPR Loss')

    _ax1.set_xlabel('Epoch', fontsize=10)
    _ax1.set_ylabel('Training Loss', color='#1f77b4', fontsize=10)
    _ax1.tick_params(axis='y', labelcolor='#1f77b4')
    _ax1.grid(True, linestyle='--', alpha=0.4)

    if _val_ndcg:
        _ax2 = _ax1.twinx()
        _x_val, _y_val = zip(*_val_ndcg)
        _ax2.plot(_x_val, _y_val, color='#ff7f0e', lw=2.2, marker='o', ms=4, label='Validation NDCG@20')
        _ax2.set_ylabel('Validation NDCG@20', color='#ff7f0e', fontsize=10)
        _ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        if 'sports' in BASELINE_REF:
            _ax2.axhline(BASELINE_REF['sports']['NDCG@20'], color='gray', ls=':', label='Baseline STAIR')
        if 'sports' in V4_REF:
            _ax2.axhline(V4_REF['sports']['NDCG@20'], color='#2ca02c', ls='--', label='GD5-v4 N-CSE')
        if 'sports' in V6_REF:
            _ax2.axhline(V6_REF['sports']['NDCG@20'], color='#9467bd', ls='-.', label='GD5-v6 P-BCSR')

    plt.title('Amazon Sports — Learning Dynamics (STAIR5-v7 UCR-D)', fontsize=12, fontweight='bold')
    _fig.tight_layout()
    plt.savefig('/kaggle/working/reports/learning_curve_sports.png')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 6d 🎯 Phân tích Động lực Học Liều lượng & Xung đột UCR-D — Amazon Sports
    - Giám sát độ khả thi của liều lượng: Biểu diễn liều lượng mục tiêu $\rho_t$ so với liều lượng thực tế đạt được $\rho_{\mathrm{eff}}$.
    - Giám sát tải xung đột $Z_t$ và hệ số suy giảm $\theta_t$ theo epoch.
    """)
    return


@app.cell
def _(STAIR5_V7_CONFIGS, parse_telemetry_jsonl, plt):
    # Cell 6d: UCR-D Dynamic Dose & Conflict Telemetry — Amazon Sports
    _sports_art = STAIR5_V7_CONFIGS['sports']['artifact_dir']
    _telemetry_sports = parse_telemetry_jsonl(_sports_art)

    plt.figure(figsize=(10, 4.5), dpi=150)
    if _telemetry_sports:
        _eps = [r['epoch'] for r in _telemetry_sports if 'diagnostics' in r and r['diagnostics']]
        _rho_eff = [r['diagnostics'].get('rho_eff', 0.0) for r in _telemetry_sports if 'diagnostics' in r and r['diagnostics']]
        _rho_t = [r['diagnostics'].get('rho_t', 0.0) for r in _telemetry_sports if 'diagnostics' in r and r['diagnostics']]
        _Z_t = [r['diagnostics'].get('Z_t', 0.0) for r in _telemetry_sports if 'diagnostics' in r and r['diagnostics']]

        if _eps:
            plt.plot(_eps, _rho_t, color='#1f77b4', lw=2.0, ls='--', label='Target Dose ρ_t')
            plt.plot(_eps, _rho_eff, color='#2ca02c', lw=2.2, label='Achieved Dose ρ_eff')
            plt.plot(_eps, _Z_t, color='#d62728', lw=1.5, alpha=0.7, label='Conflict Load Z_t')
            plt.title('Amazon Sports — UCR-D Dose Feasibility & Conflict Trajectory', fontsize=12, fontweight='bold')
            plt.xlabel('Epoch', fontsize=10)
            plt.ylabel('Metric Value', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(frameon=True)
        else:
            plt.text(0.5, 0.5, 'Chưa có telemetry UCR-D (Chế độ Fast-Path hoặc Rho=0)', ha='center', va='center')
    else:
        plt.text(0.5, 0.5, 'Chưa có telemetry training', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('/kaggle/working/reports/ucr_dynamics_sports.png')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 7a 🏋️ Huấn luyện Pha B — Amazon Baby (Kiểm chứng An toàn STAIR5-v7 / UCR-D)
    - Huấn luyện STAIR5-v7 trên tập **Amazon Baby** (19,445 Users, 7,050 Items).
    - Kiểm tra tính ổn định trên tập mật độ thưa cao với siêu tham số $\gamma=0.1, wd=0.3$.
    """)
    return


@app.cell
def _(
    BASELINE_REF,
    STAIR5_V7_CONFIGS,
    STAIR5_V7_RESULTS,
    TRACKED_METRICS,
    V4_REF,
    V6_REF,
    extract_best_validation,
    extract_test_metrics,
    run_training_stair5_v7,
):
    # Cell 7a: Training STAIR5-v7 UCR-D on Amazon Baby (Pha B: Confirmatory)
    baby_time = run_training_stair5_v7('baby')
    ep_baby, metrics_baby = extract_test_metrics(STAIR5_V7_CONFIGS['baby']['log'], STAIR5_V7_CONFIGS['baby']['artifact_dir'])
    val_ep_baby, val_m_baby = extract_best_validation(STAIR5_V7_CONFIGS['baby']['log'])
    STAIR5_V7_RESULTS['baby'] = {
        'metrics': metrics_baby,
        'best_epoch': ep_baby,
        'val_metrics': val_m_baby,
        'train_time': baby_time,
    }

    print('\n' + '=' * 80)
    print('📊 KẾT QUẢ THỰC NGHIỆM PHA B — AMAZON BABY (STAIR5-v7 / UCR-D):')
    print(f"  * Checkpoint tối ưu tại Epoch: {ep_baby} (Validation NDCG@20 = {val_m_baby.get('NDCG@20', 0.0):.4f})")
    for _m in TRACKED_METRICS:
        _val = metrics_baby.get(_m, 0.0)
        _base = BASELINE_REF['baby'].get(_m, 0.0)
        _v4_val = V4_REF['baby'].get(_m, 0.0)
        _v6_val = V6_REF['baby'].get(_m, 0.0)
        _d_base = ((_val - _base) / _base) * 100 if _base > 0 else 0.0
        _d_v4 = ((_val - _v4_val) / _v4_val) * 100 if _v4_val > 0 else 0.0
        _d_v6 = ((_val - _v6_val) / _v6_val) * 100 if _v6_val > 0 else 0.0
        print(f"  * {_m:10s} : {_val:.4f}  (vs Baseline: {_d_base:+.2f}% | vs v4: {_d_v4:+.2f}% | vs v6: {_d_v6:+.2f}%)")
    print('=' * 80)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 7b ⚡ Biểu đồ Tiêu thụ Bộ nhớ Tensor Mô hình — Amazon Baby
    - Phân tích chi tiết mức tiêu thụ VRAM thực tế trên Amazon Baby.
    """)
    return


@app.cell
def _(STAIR5_V7_CONFIGS, os, parse_telemetry_jsonl, plt, vram_profile):
    # Cell 7b: Model Tensor VRAM Profile — Amazon Baby (Paper Standard)
    _baby_art = STAIR5_V7_CONFIGS['baby']['artifact_dir']
    _telemetry_baby = parse_telemetry_jsonl(_baby_art)

    plt.figure(figsize=(9, 4.5), dpi=150)
    if _telemetry_baby:
        _eps = [r['epoch'] for r in _telemetry_baby]
        _vram_mb = [r.get('max_memory_allocated_bytes', 0) / (1024**2) for r in _telemetry_baby]
        plt.plot(_eps, _vram_mb, color='#1f77b4', lw=2.2, marker='o', ms=4, label='Peak Allocated VRAM (MB)')
        plt.title('Amazon Baby — Peak Model Tensor VRAM Profile (STAIR5-v7)', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Allocated VRAM (MB)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=True)
    elif 'baby' in vram_profile and len(vram_profile['baby']) > 0:
        plt.plot(vram_profile['baby'], color='#1f77b4', lw=1.8, label='Host NVML GPU Used (MB)')
        plt.title('Amazon Baby — Host GPU Memory Profile (NVML)', fontsize=12, fontweight='bold')
        plt.xlabel('Polling Step (~2s)', fontsize=10)
        plt.ylabel('Memory (MB)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=True)
    else:
        plt.text(0.5, 0.5, 'Chưa có dữ liệu VRAM hoặc NVML không khả dụng', ha='center', va='center')

    plt.tight_layout()
    os.makedirs('/kaggle/working/reports', exist_ok=True)
    plt.savefig('/kaggle/working/reports/vram_profile_baby.png')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 7c 📈 Động Lực Học & Quá Trình Hội Tụ — Amazon Baby
    - Trực quan hóa đường cong hàm mất mát BPR và đường cong đánh giá Validation NDCG@20 trên Amazon Baby.
    """)
    return


@app.cell
def _(
    BASELINE_REF,
    STAIR5_V7_CONFIGS,
    V4_REF,
    V6_REF,
    parse_telemetry_jsonl,
    parse_training_losses,
    parse_valid_metric,
    plt,
):
    # Cell 7c: Learning Dynamics & Convergence Profiles — Amazon Baby
    _fig, _ax1 = plt.subplots(figsize=(10, 4.8), dpi=150)

    _baby_art = STAIR5_V7_CONFIGS['baby']['artifact_dir']
    _telemetry_baby = parse_telemetry_jsonl(_baby_art)
    _log_baby = STAIR5_V7_CONFIGS['baby']['log']
    _bpr_b, _cl_b, _tot_b = parse_training_losses(_log_baby, _telemetry_baby)
    _val_ndcg_b = parse_valid_metric(_log_baby, 'NDCG@20')

    if _tot_b:
        _x_tot, _y_tot = zip(*_tot_b)
        _ax1.plot(_x_tot, _y_tot, color='#1f77b4', lw=2.0, label='Total Loss (BPR + λ·NLGCL)')
    if _bpr_b:
        _x_bpr, _y_bpr = zip(*_bpr_b)
        _ax1.plot(_x_bpr, _y_bpr, color='#1f77b4', lw=1.5, ls='--', alpha=0.7, label='BPR Loss')

    _ax1.set_xlabel('Epoch', fontsize=10)
    _ax1.set_ylabel('Training Loss', color='#1f77b4', fontsize=10)
    _ax1.tick_params(axis='y', labelcolor='#1f77b4')
    _ax1.grid(True, linestyle='--', alpha=0.4)

    if _val_ndcg_b:
        _ax2 = _ax1.twinx()
        _x_val, _y_val = zip(*_val_ndcg_b)
        _ax2.plot(_x_val, _y_val, color='#ff7f0e', lw=2.2, marker='o', ms=4, label='Validation NDCG@20')
        _ax2.set_ylabel('Validation NDCG@20', color='#ff7f0e', fontsize=10)
        _ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        if 'baby' in BASELINE_REF:
            _ax2.axhline(BASELINE_REF['baby']['NDCG@20'], color='gray', ls=':', label='Baseline STAIR')
        if 'baby' in V4_REF:
            _ax2.axhline(V4_REF['baby']['NDCG@20'], color='#2ca02c', ls='--', label='GD5-v4 N-CSE')
        if 'baby' in V6_REF:
            _ax2.axhline(V6_REF['baby']['NDCG@20'], color='#9467bd', ls='-.', label='GD5-v6 P-BCSR')

    plt.title('Amazon Baby — Learning Dynamics (STAIR5-v7 UCR-D)', fontsize=12, fontweight='bold')
    _fig.tight_layout()
    plt.savefig('/kaggle/working/reports/learning_curve_baby.png')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 7d 🎯 Phân tích Động lực Học Liều lượng & Xung đột UCR-D — Amazon Baby
    - Giám sát độ khả thi của liều lượng: $\rho_t$ vs $\rho_{\mathrm{eff}}$ và tải xung đột $Z_t$ trên Amazon Baby.
    """)
    return


@app.cell
def _(STAIR5_V7_CONFIGS, parse_telemetry_jsonl, plt):
    # Cell 7d: UCR-D Dynamic Dose & Conflict Telemetry — Amazon Baby
    _baby_art = STAIR5_V7_CONFIGS['baby']['artifact_dir']
    _telemetry_baby = parse_telemetry_jsonl(_baby_art)

    plt.figure(figsize=(10, 4.5), dpi=150)
    if _telemetry_baby:
        _eps = [r['epoch'] for r in _telemetry_baby if 'diagnostics' in r and r['diagnostics']]
        _rho_eff = [r['diagnostics'].get('rho_eff', 0.0) for r in _telemetry_baby if 'diagnostics' in r and r['diagnostics']]
        _rho_t = [r['diagnostics'].get('rho_t', 0.0) for r in _telemetry_baby if 'diagnostics' in r and r['diagnostics']]
        _Z_t = [r['diagnostics'].get('Z_t', 0.0) for r in _telemetry_baby if 'diagnostics' in r and r['diagnostics']]

        if _eps:
            plt.plot(_eps, _rho_t, color='#1f77b4', lw=2.0, ls='--', label='Target Dose ρ_t')
            plt.plot(_eps, _rho_eff, color='#2ca02c', lw=2.2, label='Achieved Dose ρ_eff')
            plt.plot(_eps, _Z_t, color='#d62728', lw=1.5, alpha=0.7, label='Conflict Load Z_t')
            plt.title('Amazon Baby — UCR-D Dose Feasibility & Conflict Trajectory', fontsize=12, fontweight='bold')
            plt.xlabel('Epoch', fontsize=10)
            plt.ylabel('Metric Value', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(frameon=True)
        else:
            plt.text(0.5, 0.5, 'Chưa có telemetry UCR-D (Chế độ Fast-Path hoặc Rho=0)', ha='center', va='center')
    else:
        plt.text(0.5, 0.5, 'Chưa có telemetry training', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('/kaggle/working/reports/ucr_dynamics_baby.png')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 8a 🏋️ Huấn luyện Pha C — Amazon Electronics (~1.7M Tương tác, Tối ưu Hóa Blackwell & Cổng kiểm soát)
    - Kiểm chứng khả năng mở rộng (Scalability) của STAIR5-v7 trên tập quy mô lớn **Amazon Electronics** (192,403 Users, 63,001 Items, 1.69M Interactions).
    - Sử dụng cấu hình chuẩn: $\gamma = 0.4, \text{batch\_size} = 4096, \text{cf\_block\_size} = 64$.
    - Tự động nhận diện GPU Blackwell 96 GB để mở rộng `cl_chunk_size` và `pair_chunk_size`.
    """)
    return


@app.cell
def _(
    BASELINE_REF,
    STAIR5_V7_CONFIGS,
    STAIR5_V7_RESULTS,
    TRACKED_METRICS,
    V4_REF,
    V6_REF,
    extract_best_validation,
    extract_test_metrics,
    run_training_stair5_v7,
):
    # Cell 8a: Training STAIR5-v7 UCR-D on Amazon Electronics (Pha C: Scale-up)
    electronics_time = run_training_stair5_v7('electronics')
    ep_elec, metrics_elec = extract_test_metrics(STAIR5_V7_CONFIGS['electronics']['log'], STAIR5_V7_CONFIGS['electronics']['artifact_dir'])
    val_ep_elec, val_m_elec = extract_best_validation(STAIR5_V7_CONFIGS['electronics']['log'])
    STAIR5_V7_RESULTS['electronics'] = {
        'metrics': metrics_elec,
        'best_epoch': ep_elec,
        'val_metrics': val_m_elec,
        'train_time': electronics_time,
    }

    print('\n' + '=' * 80)
    print('📊 KẾT QUẢ THỰC NGHIỆM PHA C — AMAZON ELECTRONICS (STAIR5-v7 / UCR-D):')
    print(f"  * Checkpoint tối ưu tại Epoch: {ep_elec} (Validation NDCG@20 = {val_m_elec.get('NDCG@20', 0.0):.4f})")
    for _m in TRACKED_METRICS:
        _val = metrics_elec.get(_m, 0.0)
        _base = BASELINE_REF['electronics'].get(_m, 0.0)
        _v4_val = V4_REF['electronics'].get(_m, 0.0)
        _v6_val = V6_REF['electronics'].get(_m, 0.0)
        _d_base = ((_val - _base) / _base) * 100 if _base > 0 else 0.0
        _d_v4 = ((_val - _v4_val) / _v4_val) * 100 if _v4_val > 0 else 0.0
        _d_v6 = ((_val - _v6_val) / _v6_val) * 100 if _v6_val > 0 else 0.0
        print(f"  * {_m:10s} : {_val:.4f}  (vs Baseline: {_d_base:+.2f}% | vs v4: {_d_v4:+.2f}% | vs v6: {_d_v6:+.2f}%)")
    print('=' * 80)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 8b ⚡ Biểu đồ Tiêu thụ Bộ nhớ Tensor Mô hình — Amazon Electronics (Kiểm chứng Cổng $<800\text{ MiB}$)
    - Phân tích chi tiết mức tiêu thụ VRAM thực tế trên Amazon Electronics.
    - Vẽ đường kiểm soát ngưỡng $800\text{ MiB}$ để minh chứng thỏa mãn điều kiện phần cứng.
    """)
    return


@app.cell
def _(STAIR5_V7_CONFIGS, os, parse_telemetry_jsonl, plt, vram_profile):
    # Cell 8b: Model Tensor VRAM Profile — Amazon Electronics (Memory Gate Verification)
    _elec_art = STAIR5_V7_CONFIGS['electronics']['artifact_dir']
    _telemetry_elec = parse_telemetry_jsonl(_elec_art)

    plt.figure(figsize=(9, 4.5), dpi=150)
    if _telemetry_elec:
        _eps = [r['epoch'] for r in _telemetry_elec]
        _vram_mb = [r.get('max_memory_allocated_bytes', 0) / (1024**2) for r in _telemetry_elec]
        _peak_val = max(_vram_mb)
        plt.plot(_eps, _vram_mb, color='#2ca02c', lw=2.0, label=f'Peak Allocated VRAM (Max: {_peak_val:.1f} MB)')
        plt.axhline(800.0, color='red', ls='--', lw=1.8, label='Execution Gate Threshold (800 MB)')
        plt.title('Amazon Electronics — Peak Model Tensor VRAM Profile (STAIR5-v7 Gate Check)', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Allocated VRAM (MB)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=True)
    elif 'electronics' in vram_profile and len(vram_profile['electronics']) > 0:
        plt.plot(vram_profile['electronics'], color='#2ca02c', lw=1.8, label='Host NVML GPU Used (MB)')
        plt.axhline(800.0, color='red', ls='--', lw=1.8, label='Execution Gate Threshold (800 MB)')
        plt.title('Amazon Electronics — Host GPU Memory Profile (NVML)', fontsize=12, fontweight='bold')
        plt.xlabel('Polling Step (~2s)', fontsize=10)
        plt.ylabel('Memory (MB)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=True)
    else:
        plt.text(0.5, 0.5, 'Chưa có dữ liệu VRAM hoặc NVML không khả dụng', ha='center', va='center')

    plt.tight_layout()
    os.makedirs('/kaggle/working/reports', exist_ok=True)
    plt.savefig('/kaggle/working/reports/vram_profile_electronics.png')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 8c 📈 Động Lực Học & Quá Trình Hội Tụ — Amazon Electronics
    - Trực quan hóa đường cong hàm mất mát BPR và đường cong đánh giá Validation NDCG@20 trên Amazon Electronics.
    """)
    return


@app.cell
def _(
    BASELINE_REF,
    STAIR5_V7_CONFIGS,
    V4_REF,
    V6_REF,
    parse_telemetry_jsonl,
    parse_training_losses,
    parse_valid_metric,
    plt,
):
    # Cell 8c: Learning Dynamics & Convergence Profiles — Amazon Electronics
    _fig, _ax1 = plt.subplots(figsize=(10, 4.8), dpi=150)

    _elec_art = STAIR5_V7_CONFIGS['electronics']['artifact_dir']
    _telemetry_elec = parse_telemetry_jsonl(_elec_art)
    _log_elec = STAIR5_V7_CONFIGS['electronics']['log']
    _bpr_e, _cl_e, _tot_e = parse_training_losses(_log_elec, _telemetry_elec)
    _val_ndcg_e = parse_valid_metric(_log_elec, 'NDCG@20')

    if _tot_e:
        _x_tot, _y_tot = zip(*_tot_e)
        _ax1.plot(_x_tot, _y_tot, color='#1f77b4', lw=2.0, label='Total Loss (BPR + λ·NLGCL)')
    if _bpr_e:
        _x_bpr, _y_bpr = zip(*_bpr_e)
        _ax1.plot(_x_bpr, _y_bpr, color='#1f77b4', lw=1.5, ls='--', alpha=0.7, label='BPR Loss')

    _ax1.set_xlabel('Epoch', fontsize=10)
    _ax1.set_ylabel('Training Loss', color='#1f77b4', fontsize=10)
    _ax1.tick_params(axis='y', labelcolor='#1f77b4')
    _ax1.grid(True, linestyle='--', alpha=0.4)

    if _val_ndcg_e:
        _ax2 = _ax1.twinx()
        _x_val, _y_val = zip(*_val_ndcg_e)
        _ax2.plot(_x_val, _y_val, color='#2ca02c', lw=2.2, marker='o', ms=4, label='Validation NDCG@20')
        _ax2.set_ylabel('Validation NDCG@20', color='#2ca02c', fontsize=10)
        _ax2.tick_params(axis='y', labelcolor='#2ca02c')
        if 'electronics' in BASELINE_REF:
            _ax2.axhline(BASELINE_REF['electronics']['NDCG@20'], color='gray', ls=':', label='Baseline STAIR')
        if 'electronics' in V4_REF:
            _ax2.axhline(V4_REF['electronics']['NDCG@20'], color='#9467bd', ls='--', label='GD5-v4 N-CSE')
        if 'electronics' in V6_REF:
            _ax2.axhline(V6_REF['electronics']['NDCG@20'], color='#ff7f0e', ls='-.', label='GD5-v6 P-BCSR')

    plt.title('Amazon Electronics — Learning Dynamics (STAIR5-v7 UCR-D)', fontsize=12, fontweight='bold')
    _fig.tight_layout()
    plt.savefig('/kaggle/working/reports/learning_curve_electronics.png')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 8d 🎯 Phân tích Động lực Học Liều lượng & Xung đột UCR-D — Amazon Electronics
    - Giám sát độ khả thi của liều lượng: $\rho_t$ vs $\rho_{\mathrm{eff}}$ và tải xung đột $Z_t$ trên Amazon Electronics.
    """)
    return


@app.cell
def _(STAIR5_V7_CONFIGS, parse_telemetry_jsonl, plt):
    # Cell 8d: UCR-D Dynamic Dose & Conflict Telemetry — Amazon Electronics
    _elec_art = STAIR5_V7_CONFIGS['electronics']['artifact_dir']
    _telemetry_elec = parse_telemetry_jsonl(_elec_art)

    plt.figure(figsize=(10, 4.5), dpi=150)
    if _telemetry_elec:
        _eps = [r['epoch'] for r in _telemetry_elec if 'diagnostics' in r and r['diagnostics']]
        _rho_eff = [r['diagnostics'].get('rho_eff', 0.0) for r in _telemetry_elec if 'diagnostics' in r and r['diagnostics']]
        _rho_t = [r['diagnostics'].get('rho_t', 0.0) for r in _telemetry_elec if 'diagnostics' in r and r['diagnostics']]
        _Z_t = [r['diagnostics'].get('Z_t', 0.0) for r in _telemetry_elec if 'diagnostics' in r and r['diagnostics']]

        if _eps:
            plt.plot(_eps, _rho_t, color='#1f77b4', lw=2.0, ls='--', label='Target Dose ρ_t')
            plt.plot(_eps, _rho_eff, color='#2ca02c', lw=2.2, label='Achieved Dose ρ_eff')
            plt.plot(_eps, _Z_t, color='#d62728', lw=1.5, alpha=0.7, label='Conflict Load Z_t')
            plt.title('Amazon Electronics — UCR-D Dose Feasibility & Conflict Trajectory', fontsize=12, fontweight='bold')
            plt.xlabel('Epoch', fontsize=10)
            plt.ylabel('Metric Value', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(frameon=True)
        else:
            plt.text(0.5, 0.5, 'Chưa có telemetry UCR-D (Chế độ Fast-Path hoặc Rho=0)', ha='center', va='center')
    else:
        plt.text(0.5, 0.5, 'Chưa có telemetry training', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('/kaggle/working/reports/ucr_dynamics_electronics.png')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 9 📊 Bảng Đối chuẩn Đa thế hệ Toàn diện (Baseline vs v4 vs v6 vs v7)
    - Bảng tổng hợp đối sánh toàn diện 4 thế hệ kiến trúc: `Recall@10`, `Recall@20`, `NDCG@10`, `NDCG@20`, và mức tăng trưởng tương đối $\Delta\% = 100 \times \frac{\text{Metric}_{v7} - \text{Metric}_{\text{ref}}}{\text{Metric}_{\text{ref}}}$.
    - Hiển thị song song qua PrettyTable và Pandas DataFrame.
    """)
    return


@app.cell
def _(
    BASELINE_REF,
    STAIR5_V7_CONFIGS,
    STAIR5_V7_RESULTS,
    TRACKED_METRICS,
    V4_REF,
    V6_REF,
    display,
    extract_test_metrics,
):
    # Cell 9: Bảng Đối chuẩn Đa thế hệ Toàn diện (STAIR Baseline vs v4 vs v6 vs v7)
    import pandas as pd
    from prettytable import PrettyTable

    _table = PrettyTable()
    _table.field_names = ['Dataset', 'Kiến trúc / Phiên bản', *TRACKED_METRICS, 'Δ vs Baseline (%)', 'Δ vs v4 (%)', 'Δ vs v6 (%)']

    _df_rows = []

    for _ds in ('sports', 'baby', 'electronics'):
        _ref_bl = BASELINE_REF[_ds]['NDCG@20']
        _ref_v4 = V4_REF[_ds]['NDCG@20']
        _ref_v6 = V6_REF[_ds]['NDCG@20']

        # 1. Baseline
        _table.add_row([_ds, 'STAIR (baseline)', *[f"{BASELINE_REF[_ds][_m]:.4f}" for _m in TRACKED_METRICS], '-', '-', '-'])
        _df_rows.append({'Dataset': _ds, 'Model': 'STAIR (baseline)', **{_m: BASELINE_REF[_ds][_m] for _m in TRACKED_METRICS}, 'Δ vs Baseline': 0.0, 'Δ vs v4': 0.0})

        # 2. v4
        _d_v4_bl = 100 * (_ref_v4 - _ref_bl) / _ref_bl
        _table.add_row([_ds, 'GD5-v4 (comparator)', *[f"{V4_REF[_ds][_m]:.4f}" for _m in TRACKED_METRICS], f"{_d_v4_bl:+.2f}", '-', '-'])
        _df_rows.append({'Dataset': _ds, 'Model': 'GD5-v4 (comparator)', **{_m: V4_REF[_ds][_m] for _m in TRACKED_METRICS}, 'Δ vs Baseline': _d_v4_bl, 'Δ vs v4': 0.0})

        # 3. v6
        _d_v6_bl = 100 * (_ref_v6 - _ref_bl) / _ref_bl
        _d_v6_v4 = 100 * (_ref_v6 - _ref_v4) / _ref_v4
        _table.add_row([_ds, 'GD5-v6 (BCSR)', *[f"{V6_REF[_ds][_m]:.4f}" for _m in TRACKED_METRICS], f"{_d_v6_bl:+.2f}", f"{_d_v6_v4:+.2f}", '-'])
        _df_rows.append({'Dataset': _ds, 'Model': 'GD5-v6 (BCSR)', **{_m: V6_REF[_ds][_m] for _m in TRACKED_METRICS}, 'Δ vs Baseline': _d_v6_bl, 'Δ vs v4': _d_v6_v4})

        # 4. v7 (Lấy từ STAIR5_V7_RESULTS hoặc đọc từ đĩa nếu chạy riêng rẽ)
        _res_entry = STAIR5_V7_RESULTS.get(_ds)
        if not _res_entry:
            _art = STAIR5_V7_CONFIGS[_ds]['artifact_dir']
            _log = STAIR5_V7_CONFIGS[_ds]['log']
            _ep, _m = extract_test_metrics(_log, _art)
            if _m:
                _res_entry = {'metrics': _m, 'best_epoch': _ep}

        _cur_res = _res_entry.get('metrics', {}) if _res_entry else {}
        _cur_res_norm = {}
        if _cur_res:
            for _k, _v in _cur_res.items():
                for _tm in TRACKED_METRICS:
                    if _k.lower() == _tm.lower():
                        _cur_res_norm[_tm] = _v
        if _cur_res_norm and all(_m in _cur_res_norm for _m in TRACKED_METRICS):
            _d_bl = 100 * (_cur_res_norm['NDCG@20'] - _ref_bl) / _ref_bl
            _d_v4 = 100 * (_cur_res_norm['NDCG@20'] - _ref_v4) / _ref_v4
            _d_v6 = 100 * (_cur_res_norm['NDCG@20'] - _ref_v6) / _ref_v6
            _table.add_row([_ds, 'STAIR5-v7 (UCR-D)', *[f"{_cur_res_norm[_m]:.4f}" for _m in TRACKED_METRICS], f"{_d_bl:+.2f}", f"{_d_v4:+.2f}", f"{_d_v6:+.2f}"] )
            _df_rows.append({'Dataset': _ds, 'Model': 'STAIR5-v7 (UCR-D)', **{_m: _cur_res_norm[_m] for _m in TRACKED_METRICS}, 'Δ vs Baseline': _d_bl, 'Δ vs v4': _d_v4})
        else:
            _table.add_row([_ds, 'STAIR5-v7 (UCR-D)', *['not measured'] * 4, '-', '-', '-'])

    print(_table)
    results_df = pd.DataFrame(_df_rows)
    display(results_df) if 'display' in globals() else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cell 10 📦 Đóng gói Lưu trữ Toàn bộ Minh chứng, Logs, Manifests & Checkpoints
    - Nén toàn bộ kết quả, biểu đồ PNG, logs, telemetry và manifest JSON để nghiệm thu khóa luận tốt nghiệp.
    """)
    return


@app.cell
def _(
    STAIR5_V7_CONFIGS,
    STAIR5_V7_RESULTS,
    extract_test_metrics,
    json,
    os,
    shutil,
    time,
):
    # Cell 10: Đóng gói Lưu trữ Toàn bộ Minh chứng (Logs, Checkpoints, Telemetry, Reports)
    _archive_dir = '/kaggle/working/stair5_v7_artifacts'
    os.makedirs(_archive_dir, exist_ok=True)
    _target_dirs = ['/kaggle/working/logs/stair5_v7', '/kaggle/working/reports', '/kaggle/working/runs/stair5_v7', os.path.expanduser('~/logs/stair5_v7'), os.path.expanduser('~/reports')]
    for _src in _target_dirs:
        if os.path.exists(_src):
    # 1. Thu thập logs, reports và runs
            _dst = os.path.join(_archive_dir, os.path.basename(_src))
            try:
                shutil.copytree(_src, _dst, dirs_exist_ok=True)
            except Exception:
                pass
    _datasets_manifest = {}
    for _ds in ('sports', 'baby', 'electronics'):
        _res_entry = STAIR5_V7_RESULTS.get(_ds, {})
        if not _res_entry:
            _art = STAIR5_V7_CONFIGS[_ds]['artifact_dir']
            _log = STAIR5_V7_CONFIGS[_ds]['log']
            _ep, _m = extract_test_metrics(_log, _art)
            _res_entry = {'metrics': _m, 'best_epoch': _ep, 'train_time': None}
        _datasets_manifest[_ds] = {'test_metrics': _res_entry.get('metrics', {}), 'best_epoch': _res_entry.get('best_epoch', None), 'train_time_min': _res_entry.get('train_time') / 60 if _res_entry.get('train_time') else None}
    _manifest = {'project': 'STAIR5-v7 / UCR-D', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'datasets': _datasets_manifest}
    # 2. Tạo Manifest JSON tổng hợp
    _manifest_path = os.path.join(_archive_dir, 'stair5_v7_manifest.json')
    with open(_manifest_path, 'w', encoding='utf-8') as _f:
        json.dump(_manifest, _f, indent=2, ensure_ascii=False)
    _zip_path = '/kaggle/working/stair5_v7_complete_artifacts.zip'
    shutil.make_archive('/kaggle/working/stair5_v7_complete_artifacts', 'zip', _archive_dir)
    print('=' * 80)
    print('📦 ĐÓNG GÓI HOÀN TẤT THÀNH CÔNG!')
    print(f'  * Thư mục Artifacts : {_archive_dir}')
    if os.path.exists(_zip_path):
        print(f'  * File Nén Tổng Hợp : {_zip_path} ({os.path.getsize(_zip_path) / 1024 ** 2:.2f} MB)')
    # 3. Nén file zip hoàn chỉnh
    print('=' * 80)
    return


if __name__ == "__main__":
    app.run()
