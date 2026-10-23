# Paper figure attribution

The images below are from *Noise-Robust Heart Sound Analysis for Wearable Monitoring Using Cascaded Dual-Channel ANC and Wavelet Denoising* by Jiaqi Zhang, Chenghao Yang, Chen Shen, Tiancheng Cao and Hen-Wei Huang, TechRxiv preprint v1, 11 February 2026.

| File | Original figure | Content |
| --- | --- | --- |
| `figure2-subband-anc.jpg` | Figure 2 | Original subband ANC algorithm diagram. |
| `figure3-wavelet-denoising.jpg` | Figure 3 | Original wavelet decomposition, thresholding, reconstruction and sampling diagram. |
| `figure4-classifier-training.jpg` | Figure 4 | Training and validation accuracy/loss, and the test confusion matrix. |
| `figure5-snr-gain.png` | Figure 5 | SNR gain across injected SNR levels for ANC, Wavelet and ANC+Wavelet. |
| `figure6-classification-accuracy.png` | Figure 6 | Accuracy across injected SNR levels for four processing pipelines and the clean reference. |
| `figure7-clean-noisy.png` | Figure 7 | Clean and noisy PCG waveforms with zoomed S2 regions. |
| `figure8-denoising-waveforms.png` | Figure 8 | ANC, wavelet and cascaded ANC+Wavelet waveforms with zoomed S2 regions. |

- Source: https://doi.org/10.36227/techrxiv.177084814.40868704/v1
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), as stated in the supplied preprint. The included CC BY 4.0 legal text is at [LICENSE-WEIGHTS.txt](../../licenses/LICENSE-WEIGHTS.txt); its terms also apply to these separately identified figures.
- Extraction: JPEG image streams on PDF pages 4, 5 and 6 were extracted without recompression. Image objects on PDF pages 7 and 8 were decoded and saved as lossless PNG. No images were redrawn, cropped, resized or annotated. Markdown display scaling does not change the image files. The source PDF is not distributed.
- Provenance: [source.json](source.json) records the PDF fingerprint, embedded image identities, original captions, dimensions and extracted-file fingerprints.

Figures 2 and 3 are reproduced as original algorithm schematics, with all labels and connections preserved. They are not redrawn as code-exact implementation diagrams; the repository configuration remains documented in the main README and `configs/experiment.json`.

Figures 4, 5 and 6 preserve the result plots in the supplied preprint, including their original axes, legends, curves and confusion-matrix counts. They are separate from the current reproduction measurements stored in `results/measured/`; the main README reports the baseline accuracy and sample-count differences.

Figures 7 and 8 are qualitative waveform illustrations from the preprint. They illustrate the source paper; current reproduction metrics are stored in `results/measured/`. No unreported source recording, injected SNR or wavelet setting is assigned to these original images.

These figures retain CC BY 4.0 and are excluded from the repository's MIT scope. Retain attribution, link the license and indicate changes when distributing adaptations.
