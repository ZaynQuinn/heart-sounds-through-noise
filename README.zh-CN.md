# 在噪声中听见心跳

[English](README.md) | **简体中文**

**面向心音分类的双通道 ANC 与小波级联降噪。**

[TechRxiv 预印本](https://doi.org/10.36227/techrxiv.177084814.40868704/v1) · [EMBC 2026 会议报告](https://cmsworkshops.com/EMBC2026/view_paper.php?PaperNum=4350&SessionID=1053)

[Pipeline](#pipeline) · [实验结果](#实验结果) · [如何复现](#如何复现)

## TL;DR

可穿戴设备采集的心音容易受到环境噪声干扰。本项目结合双通道自适应噪声抵消（ANC）、小波降噪和仅含 622 个参数的分类器，对含噪心音记录（phonocardiogram，PCG）进行正常与异常分类。

- **方法：** 双通道子带 ANC → 小波降噪 → log-Mel 特征 → 622 参数 MicroHeartCNN。所有降噪分支使用同一个分类器。
- **实验：** PhysioNet/CinC 2016 心音与 MUSAN 噪声，2 kHz 采样；486 段测试录音切分为 6,737 个片段，在 −10～20 dB 七档 SNR 下比较六个分支。
- **结果：** 干净输入准确率为 **83.84%**。在 −10 dB 下，ANC → Wavelet D1+D2 将准确率从 **55.87% 提升至 68.58%**，增加 **12.71 个百分点**。

仓库包含预处理、训练、评价与绘图代码，以及基准权重、固定数据划分和混噪配方。

## Pipeline

![心音处理流程：双通道混噪、子带 ANC、小波降噪、log-Mel 特征提取与分类](assets/diagrams/heart-sound-pipeline.png)

实验由同一段心音和噪声合成主通道与参考通道。主通道使用七档输入 SNR，参考通道固定为 −20 dB，两个通道均含心音。混噪和降噪在整段录音上完成，再切分为分类片段。

分类器在未人工加入 MUSAN 噪声的训练集上训练，评价时固定权重。六个对比分支为 **Noisy、ANC、Wavelet D1、Wavelet D1+D2、Cascade D1、Cascade D1+D2**。Cascade 表示先 ANC 后小波；D1、D2 分别是最高频和次高频细节层。

### 降噪算法如何工作

ANC 用参考通道估计相关噪声，再从主通道中减去；残差信号用于更新滤波器。小波将 ANC 输出分解到不同尺度，对选中的细节系数做软阈值处理，其余系数保留，再重建波形。

**子带 ANC**

![子带 ANC 算法图](assets/paper/figure2-subband-anc.jpg)

**小波分解、阈值处理与重建**

![小波降噪算法：分解、阈值处理、重建与采样示意](assets/paper/figure3-wavelet-denoising.jpg)

### 降噪前后的波形

心音中的 S1、S2 分别对应第一、第二心音。下图对比干净心音与混噪心音：噪声掩盖了较弱的瞬态细节，右侧放大的 S2 区域尤其明显。

![干净与混噪心音对比，右侧放大 S2 区域](assets/paper/figure7-clean-noisy.png)

ANC 利用参考通道抵消相关噪声，小波再对细节子带做软阈值处理。在下方示例中，单独 ANC 保留了瞬态形态，但还有残余噪声；单独小波降低了背景噪声，也平滑了 S2 的细节。级联处理在降低噪声的同时保留了较清晰的 S2 波形。

![ANC、小波及级联降噪的波形对比，右侧放大 S2 区域](assets/paper/figure8-denoising-waveforms.png)

### 按 SNR 选择小波子带

**ANC + Wavelet 级联方案**按输入 SNR 选择软阈值处理的细节子带：

| 本实验输入 SNR | ANC 后的小波处理 | 对应评价分支 |
|---|---|---|
| −10、−5、0、5 dB | D1+D2 软阈值处理 | `cascade_d1d2` |
| 10、15、20 dB | 仅 D1 软阈值处理 | `cascade_d1` |

选择依据是混噪实验中设定的输入 SNR。全量评价保留两个固定配置的结果，分档级联结果按上表读取对应分支。

<details>
<summary>完整信号处理参数</summary>

**分窗与特征。** 音频转为 2,000 Hz 单声道；每个分类片段为 6,000 点，步长为 3,000 点。短于 3 秒的录音在右侧补零，长录音末尾不足 3 秒的部分舍弃。片段编号、标签和数据划分使用 `manifests/segments.csv`。

| 参数 | 值 |
|---|---|
| STFT | `n_fft=256`、`win_length=256`、周期型 `hann` 窗、`hop_length=64` |
| 边界处理 | `center=True`、`pad_mode="constant"`，零填充 |
| Mel 频谱 | `power=2.0`、`n_mels=64`、`fmin=20`、`fmax=400` |
| Mel 滤波器 | `htk=False`、`norm="slaney"`、`dtype=float32` |
| 对数 | `np.log(mel + 1e-6)`，自然对数，输出 float32 |
| 标准化 | 每个 Mel 频带使用训练集统计量计算 `(feature - mean) / std` |

**混噪。** `manifests/noise_mixing.csv` 固定噪声文件、起点、截取或循环方式及目标 SNR。噪声不足整段心音长度时循环拼接。对心音 `s`、噪声 `n` 和目标 SNR `q`，以 float64 计算：

```text
EPS = 1e-12
rms(x) = sqrt(mean(x*x) + EPS)
alpha = rms(s) / (rms(n) * 10**(q/20) + EPS)
mixed = float32(s + alpha*n)
```

主通道 SNR 为 −10、−5、0、5、10、15、20 dB，参考通道为 −20 dB。两通道分别缩放；若峰值超过 0.98，则整体等比例缩小，再保存为 PCM16。

**ANC 与小波。** 低于 250 Hz 和高于 900 Hz 的主通道分量直通；250–400 Hz、400–900 Hz 分别做自适应滤波。ANC 滤波器长度为 512，两个处理频带的步长分别为 0.005、0.01，`leak=1e-5`、`delta=1e-6`，延迟对齐最大搜索范围为 120 点。小波使用 sym6、4 层分解、对称延拓、通用软阈值，阈值系数为 1.2，重建输出限幅为 ±0.98。D1 和 D1+D2 分别处理最高一个、两个细节层。参数保存在 `configs/experiment.json`。

</details>

## 实验结果

### 降噪增益

![ANC、小波与级联方案在不同输入 SNR 下的降噪增益](assets/paper/figure5-snr-gain.png)

纵轴 ΔSNR 表示处理后的 SNR 相比输入提高了多少。低 SNR 下，级联方案的增益约为 3.8–4.1 dB；输入越干净，可进一步改善的空间越小。

### 噪声下的分类表现

![Noisy、Wavelet、ANC 与 ANC + Wavelet 在不同 SNR 下的分类准确率](assets/paper/figure6-classification-accuracy.png)

深蓝色菱形曲线为 **ANC + Wavelet**，虚线为干净输入参考。级联方案在各档 SNR 下均取得最高准确率，低 SNR 下的改善更明显。

### 分类器训练

![MicroHeartCNN 的训练与验证曲线及测试混淆矩阵](assets/paper/figure4-classifier-training.jpg)

左侧展示 50 轮训练中的准确率与损失，右侧展示正常、异常心音的分类结果。

<details>
<summary>实验配置</summary>

| 项目 | 设置 |
|---|---|
| 训练 / 验证 / 测试录音 | 2,268 / 486 / 486 |
| 训练 / 验证 / 测试片段 | 30,664 / 6,277 / 6,737 |
| 训练配置 | 50 epochs、batch size 32、学习率 0.001、seed 42；按验证集选择最佳模型 |
| 固定分类器 | `heart_micro_20251230_184645`，622 个参数 |
| 干净输入结果 | Accuracy **83.8355%**，Macro F1 **81.6308%** |

</details>

<details>
<summary>复现实测准确率</summary>

| 输入 SNR | Noisy | ANC → D1 | ANC → D1+D2 | 分档级联 |
|---:|---:|---:|---:|---:|
| −10 dB | 55.87% | 60.56% | 68.58% | **68.58%** |
| −5 dB | 61.56% | 65.99% | 72.11% | **72.11%** |
| 0 dB | 65.85% | 71.09% | 75.17% | **75.17%** |
| 5 dB | 70.42% | 75.51% | 76.71% | **76.71%** |
| 10 dB | 73.61% | 79.10% | 77.84% | **79.10%** |
| 15 dB | 77.38% | 81.28% | 78.45% | **81.28%** |
| 20 dB | 79.84% | 82.43% | 79.12% | **82.43%** |

D1+D2 级联在低 SNR 下提升较大，D1 级联在 10～20 dB 下更好；20 dB 时，固定 D1+D2 的准确率略低于 Noisy。全部六分支指标保存在 `results/measured/full_denoising_metrics.json`。

</details>

### 与论文的对应关系

| 项目 | 论文 | 本仓库 |
|---|---|---|
| 干净输入准确率 | 正文 83.12% | 固定权重评价 83.8355% |
| 测试片段数 | 图 4 共 6,453 个 | 6,737 个 |

## 如何复现

按 **准备环境 → 准备数据 → 干净基准 → 单录音试跑 → 全量评价** 运行。基准权重已包含在仓库中，可以直接评价。

### 1. 准备环境

下载或解压项目后，在包含 `README.md` 的项目根目录打开终端。已验证环境为 **macOS ARM64、CPU、Python 3.9.6、PyTorch 2.8.0、NumPy 1.26.4、librosa 0.11.0**。以下命令使用 macOS/Linux shell，`python3` 应指向 Python 3.9 解释器。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check

DATA="/path/to/input_data"
RUN="/path/to/new_run"
```

将占位路径替换为实际的绝对路径。`DATA` 保存输入音频，`RUN` 保存实验输出，两者互不包含。首次下载时 `DATA` 必须尚不存在；每次实验使用新的 `RUN` 路径。输出子目录和结果文件由脚本创建，后续命令在同一个已激活的终端中执行。

`requirements.txt` 包含预处理、降噪、训练、评价和绘图所需的依赖。安装一次即可，后面的可选训练步骤无需另装依赖。

### 2. 准备数据

**心音：** 使用 [PhysioNet/CinC Challenge 2016 v1.0.0](https://physionet.org/content/challenge-2016/1.0.0/) 的 training-a～f。运行以下命令取得全部 3,240 个 WAV，并核对文件大小与 SHA-256：

```sh
python -m scripts.download_clean --output-root "$DATA"
python -m scripts.verify_inputs --data-root "$DATA" --kind clean
```

下载中断时，先等待原进程退出，再运行 `python -m scripts.download_clean --output-root "$DATA" --resume`。已有数据可跳过下载，按下方目录放置后直接校验。

**噪声：** 从 [MUSAN / OpenSLR 17](https://www.openslr.org/17/) 下载并解压语料，将解压所得的 `noise` 目录复制为 `$DATA/musan_noise`。项目使用 `free-sound` 和 `sound-bible` 中的 364 段噪声。保留原始语料的许可及署名文件。

```text
input_data/
├── data/raw/
│   ├── training-a/*.wav
│   ├── training-b/*.wav
│   ├── training-c/*.wav
│   ├── training-d/*.wav
│   ├── training-e/*.wav
│   └── training-f/*.wav
└── musan_noise/
    ├── free-sound/*.wav
    └── sound-bible/*.wav
```

校验原始噪声，再生成 2 kHz 单声道版本：

```sh
python -m scripts.verify_inputs --data-root "$DATA" --kind noise
python -m scripts.prepare_noise --noise-root "$DATA/musan_noise" \
  --output-root "$DATA/musan_noise_2k"
python -m scripts.verify_inputs --data-root "$DATA" --kind noise_2k
```

`musan_noise_2k` 必须为新目录。校验失败时先核对数据版本、文件和路径。心音已通过官方重新下载验证；噪声重采样已在保留的原始 MUSAN 文件上验证，完整语料的重新下载路径尚未验证。

### 3. 复现干净测试基准

```sh
python -m scripts.prepare_features --data-root "$DATA" \
  --output-root "$RUN/clean_features" --split test
python -m scripts.evaluate --data-root "$RUN/clean_features" \
  --output "$RUN/clean.json"
```

生成 6,737 个测试片段的特征，并在 `clean.json` 中输出结果。基准 Accuracy 为 **83.8355%**，Macro F1 为 **81.6308%**。“干净”指未人工加入 MUSAN 噪声。

### 4. 单录音试跑

使用一段心音和 5 dB 条件检查完整流程：

```sh
python -m scripts.prepare_denoising --data-root "$DATA" \
  --output-root "$RUN/sample_waveforms" \
  --recordings training-f/f0015.wav --snrs 5
python -m scripts.evaluate_denoising --waveform-root "$RUN/sample_waveforms" \
  --output-dir "$RUN/sample_metrics"
```

生成主通道、参考通道及五种降噪输出，共七个 WAV。`sample_metrics/metrics.json` 和 `predictions.csv` 记录六个分类分支的指标及逐片段预测。

### 5. 全量降噪评价与绘图

```sh
python -m scripts.prepare_denoising --data-root "$DATA" \
  --output-root "$RUN/full_waveforms"
python -m scripts.evaluate_denoising --waveform-root "$RUN/full_waveforms" \
  --output-dir "$RUN/full_metrics"
python -m scripts.plot_results --metrics "$RUN/full_metrics/metrics.json" \
  --clean-result "$RUN/clean.json" --output-dir "$RUN/figures"
```

输出 **23,814 个 WAV、42 组分类指标、282,954 条预测**，以及 PNG/SVG 曲线。将 `full_metrics/metrics.json` 与仓库中的 `results/measured/full_denoising_metrics.json` 对照即可核对全量结果。

`scripts/plot_results.py` 绘制六个固定分支。汇总分档级联结果时，在 `metrics.json` 中按 `snr` 与 `branch` 取值：−10、−5、0、5 dB 读取 `cascade_d1d2`，10、15、20 dB 读取 `cascade_d1`；`accuracy` 和 `macro_f1` 为该条件的分类指标。

<details>
<summary>可选：从原始音频重新训练分类器</summary>

```sh
python -m scripts.prepare_features --data-root "$DATA" \
  --output-root "$RUN/all_features" --split all
python -m scripts.compute_normalization --data-root "$RUN/all_features" \
  --output-dir "$RUN/normalization"
python -m scripts.train --data-root "$RUN/all_features" \
  --norm_dir "$RUN/normalization" --output-dir "$RUN/training"
```

标准化仅使用训练集，统计计算约需 2 GiB 空闲内存。默认在 CPU 上训练 50 轮，结束后自动用本次最佳模型评价测试集，结果写入 `training/test_results.json`。`scripts/evaluate.py` 和 `scripts/evaluate_denoising.py` 始终使用仓库基准权重。不同环境下的训练结果可能存在差异。

</details>

## 引用与许可

Jiaqi Zhang, Chenghao Yang, Chen Shen, Tiancheng Cao, and Hen-Wei Huang. “Noise-Robust Heart Sound Analysis for Wearable Monitoring Using Cascaded Dual-Channel ANC and Wavelet Denoising.” *TechRxiv*, preprint, 11 February 2026. DOI: [10.36227/techrxiv.177084814.40868704/v1](https://doi.org/10.36227/techrxiv.177084814.40868704/v1).

```bibtex
@misc{zhang2026noiserobust,
  author       = {Zhang, Jiaqi and Yang, Chenghao and Shen, Chen and Cao, Tiancheng and Huang, Hen-Wei},
  title        = {Noise-Robust Heart Sound Analysis for Wearable Monitoring Using Cascaded Dual-Channel {ANC} and Wavelet Denoising},
  year         = {2026},
  month        = feb,
  howpublished = {TechRxiv},
  note         = {Preprint, version 1},
  doi          = {10.36227/techrxiv.177084814.40868704/v1},
  url          = {https://doi.org/10.36227/techrxiv.177084814.40868704/v1}
}
```

数据来源：PhysioNet/CinC 2016，Liu et al., 2016，[DOI: 10.1088/0967-3334/37/12/2181](https://doi.org/10.1088/0967-3334/37/12/2181)；MUSAN，[OpenSLR 17](https://www.openslr.org/17/)。

代码、项目文档与项目示意图采用 [MIT](LICENSE)；基准权重采用 [CC BY 4.0](licenses/LICENSE-WEIGHTS.txt)。算法图、实验结果图与波形图取自上述作者的 TechRxiv 预印本图 2–8，沿用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。数据及依赖遵循各自许可。

[作者署名](docs/AUTHORS.md) · [许可范围](docs/LICENSING.md)
