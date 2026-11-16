# 2D elasticity with learned local deformation bases

Run the original **120,626,288-DOF smooth-density cantilever** with two frozen FNO models, GPU reduced assembly and **three geometric multigrid PCG corrections**. This is the original V9 benchmark with displacement L2 error approximately **0.06700% initially and 0.00009565% after three corrections**, and cell energy-density L2 error **88.2324% and 0.85111%**. An independently converged fine-grid Q4 solution supplies the reported full-domain errors. Each run computes its own results.

## Run on Windows

The tested configuration is 64-bit Windows, MATLAB R2025a with Parallel Computing Toolbox, NVIDIA RTX 4090 (24 GB), 128 GB system RAM and the CUDA 12.1 build of PyTorch 2.5.1. Install 64-bit Python 3.10 on PATH and a compatible NVIDIA driver. The included Windows MEX binaries contain Ampere / Ada code and PTX. The build sources are provided for a different supported MATLAB / compiler configuration.

1. Extract or clone the repository.
2. Run `SETUP.cmd` once. It creates `.venv` and installs the pinned Python packages. The first installation downloads approximately 2.5 GB for PyTorch.
3. Run `RUN.cmd`. Read `results/<timestamp>/result.json` after completion. The complete MATLAB log is in the same directory.

If Python 3.10 is installed in a custom environment, run `python setup_env.py --python "C:\path\to\python.exe"`. MATLAB can be selected with `--matlab "C:\path\to\MATLAB.exe"` or the `MATLAB_EXE` environment variable.

```powershell
# Quick installation check, 33,488 DOFs.
.venv\Scripts\python.exe run.py --size small

# Default original smooth-density benchmark, 120,626,288 DOFs.
.venv\Scripts\python.exe run.py

# Use FNO only for internal substructures and solve the free-edge bases exactly.
.venv\Scripts\python.exe run.py --edge exact

# Smaller smooth-density example, 11,902,400 DOFs.
.venv\Scripts\python.exe run.py --size 10m

# Previous independent fine-cell random-density case, 11,902,400 DOFs.
.venv\Scripts\python.exe run.py --case iid --size 10m

# Write the full material, initial displacement, three-step displacement and reference.
.venv\Scripts\python.exe run.py --save-fields

# Run only the AI initializer and three corrections, without computing the reference.
.venv\Scripts\python.exe run.py --skip-reference
```

The default has **10,981 by 5,491 unit-square Q4 elements**, **1,220 by 610 substructures**, unit thickness, Poisson ratio 0.3 and a left wall fixed in both displacement directions. The downward unit point force is at the right mid-edge node **(10,981, 2,745)**. Its one-based loaded node is **60,310,398**, and its y displacement DOF is **120,620,796**. Both nonuniform internal and free-edge substructures use their frozen FNO models; clamped and corner substructures are solved exactly. The physical boundary, full-grid assembly, three corrections and frozen model hashes match the original optimized V9 benchmark.

At zero-based cell coordinates `(i,j)`, the default material is sampled at cell centers as:

```text
rho(j,i) = 0.55 + 0.35*cos(3*pi*(j+0.5)/ny)*sin(4*pi*(i+0.5)/nx)
E/E0     = 1e-6 + (1-1e-6)*rho
```

`E0=1`; density ranges approximately from 0.2 to 0.9. The `small` and `10m` presets sample this same normalized formula on their own unit-square-element beams, with a right mid-edge point force. The result JSON records the actual domain, force node, modulus rule and material checksum.

`--case iid --size 10m` retains the previous independent random example: MATLAB `mt19937ar`, seed 91732, fine-cell Uniform[0.01,1] final relative moduli, `Emin=0`, physical domain 181 by 91, element side 1/19, force at (181,45), and a solid rectangle x/Lx in [32/181,78/181], y/Ly in [18/91,50/91]. It is an optional case with its own separately reported results.

## What runs

The internal FNO uses an actual 22 by 22 material window. The free-edge FNO uses an actual 16 by 22 bottom-edge window. Inputs and the resulting eight-column displacement bases are rotated between the bottom edge and the physical edge. Real structure boundaries stop the oversampling extension. Non-corner nodes on free physical edges participate in the auxiliary solve. Clamped and corner substructures use the original exact auxiliary elasticity problem. A completely uniform physical oversampling window uses its precomputed exact template.

Both models were trained independently on 100,000 blocks. Inference uses the shipped FP32 TorchScript models. The eight raw columns enter the reduced assembly directly. The model decoder retains the three 2D rigid motions. Basis construction, reduced assembly, reconstruction and Q4 corrections use the same native GPU algorithms as the source V9 example. Only the small set of numerical local bases is computed with SciPy in a separate Python process.

The reference solve starts from zero and checks the true residual against 1e-9. The reported displacement L2 covers all physical nodal vector components. Energy-density L2 covers the scalar, cell-average strain energy densities over every Q4 element. The displacement-error energy norm and the loaded-node displacement error are also recorded. The loaded-node neighbourhood is included.

`timing_seconds.seed_ready` stops when the physical AI displacement is ready on the GPU. `timing_seconds.corrected_ready` includes the three correction steps. `complete_call_seconds` additionally includes certification and host return. These are one warmed call, measured separately from material generation, the warmup problem, the reference solve, error integration and disk output. Timings depend on the hardware and software configuration. Historical and packaging-validation results are in `validation.json`.

## Direct MATLAB use and rebuilding

After `SETUP.cmd`, in the repository directory run:

```matlab
report = run_random_density;                  % Original smooth 120m case and reference.
report = run_random_density('small');         % Quick case.
report = run_random_density('10m','exact');    % Exact nonuniform free-edge bases.
```

`mechanics2d(cfg)` also accepts your own `[nely,nelx]` relative material field in `cfg.rho`, with y varying along the first array axis. Every substructure has eight coarse coefficients. The mesh sizes satisfy `nelx=9*nslx+1`, `nely=9*nsly+1`.

Rebuild with MATLAB's supported C++ and CUDA MEX toolchain:

```powershell
.venv\Scripts\python.exe run.py --size small --build
```

All five MEX files are built from the included source. `build_mex.m` locates LibTorch from the repository-local environment. If a supported external CUDA compiler is needed, set `V9_NVCC_DIR` to its `bin` directory before invoking the build. The shipped binaries were validated with MATLAB R2025a on Windows. Other operating systems require adapting and validating the MEX build.

## Files and GitHub

The repository carries the two deployment models, CUDA sources and binaries, MATLAB mechanics code, the exact local Q4 teacher and environment scripts. Python packages are installed into the ignored `.venv` directory. Generated field files and results go into the ignored `results` directory. Each distributed file is below GitHub's regular per-file limit. Run `python verify.py` to check the release files against `manifest.json`.

Upload the extracted repository contents to GitHub. The zip can also be attached to a GitHub Release. The two model hashes and original component hashes are recorded in `source_origin.json` and `gpu_basis/models/export.json`. The model `.ts` files contain the frozen inference graphs and parameters.

All five MEX files were rebuilt from the included sources and checked in a freshly installed virtual environment. The updated default was rerun: its initial displacement, three-step displacement and independent reference arrays match the saved original optimized V9 smooth 120m arrays **bit for bit**. `validation.json` records the measured errors, configuration, component hashes and comparison results. The previous independent-random validation is retained as historical evidence and identified by case. The training dataset is not needed to run the examples.

## 中文快速使用

默认算例已改为此前讨论并核验过的 **1.206 亿自由度平滑材料梁**：10,981 × 5,491 个单位正方形 Q4 单元，1,220 × 610 个子结构，左端固支，右端中点施加负 y 方向单位集中力；内部和自由边均使用冻结 FNO 模型，钳制块及角块精确计算。使用原始材料公式和 `Emin=1e-6`。该题的原始位移 L2 误差为 0.06700% → 0.00009565%，应变能密度 L2 误差为 88.2324% → 0.85111%。程序会实际计算参考解与全域误差。

解压后先运行 `SETUP.cmd`，再运行 `RUN.cmd` 即为上述默认题。`RUN.cmd --size small` 可先检查环境。此前的千万自由度逐细单元独立随机密度题保留为 `RUN.cmd --case iid --size 10m`。入口文件名和发布目录名为兼容上一版而保留。

需要 Windows、MATLAB、Parallel Computing Toolbox、Python 3.10、NVIDIA GPU 和驱动。源码和模型约 10 MB；首次安装 PyTorch 还会下载约 2.5 GB 的依赖。上传 GitHub 时使用本目录中的发布文件，`.gitignore` 已覆盖本机依赖与计算输出。
