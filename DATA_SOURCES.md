# Data sources and licences

Nothing in this repository is real factory data. The MES records, the SOP documents and the mapping of images to
units are **synthetic**. The only real data are public benchmark images, which are **not** stored in the repo
(`.gitignore` excludes images and archives); only per-image scores and dataset-relative file paths are committed.

## 1. VisA (Visual Anomaly dataset) - used

| | |
|---|---|
| Publisher | Amazon Science, "SPot-the-Difference" project |
| Page | https://github.com/amazon-science/spot-diff |
| Archive | https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar |
| Archive size / SHA-256 | 1,929,840,640 bytes / `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362` |
| Licence (verbatim, project README) | "The data is released under the CC BY 4.0 license." |
| Licence file in the archive | `LICENSE-DATASET` = full text of Creative Commons Attribution 4.0 International |
| Downloaded | 2026-10-13 |
| What is used | categories `pcb1` and `pcb4` (printed circuit boards) and the official split `split_csv/1cls.csv` |
| Local location | `D:\data\visa\raw\{pcb1,pcb4,split_csv,LICENSE-DATASET}` (about 515 MB). Override with `MESQ_VISA_ROOT`. |

Only `pcb1`, `pcb4`, `split_csv` and `LICENSE-DATASET` were extracted. VisA is published only as one archive (no
per-category files), so the whole 1.9 GB archive had to be downloaded once. It is still at
`D:\data\visa\VisA_20220922.tar` (with its listing `members.txt` and `.sha256`) and can be deleted to free space;
the SHA-256 above lets you verify a re-download.

Citation:

> Y. Zou, J. Jeong, L. Pemula, D. Zhang, O. Dabeer. *SPot-the-Difference Self-Supervised Pre-training for Anomaly
> Detection and Segmentation.* ECCV 2022. arXiv:2207.14315.

Attribution (CC BY 4.0): images (c) the VisA authors / Amazon, used unmodified for evaluation; this repository
redistributes no images. Scores and file names in `results/vision/*.csv` are derived data.

## 2. MVTec AD - licence checked, not downloaded

The task brief asked for MVTec AD. Its official page (https://www.mvtec.com/company/research/datasets/mvtec-ad,
checked 2026-10-13) states, verbatim:

> The data is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
> (CC BY-NC-SA 4.0). In particular, it is not allowed to use the dataset for commercial purposes. If you are unsure
> whether or not your application violates the non-commercial use clause of the license, please contact us.

Why it was not used: the page now delivers the dataset only after filling in a registration ("Fill in the form")
web form that asks for personal details, which this project did not do on the owner's behalf. The per-category
archive links that used to be public (`www.mydrive.ch/shares/38536/...`) return HTTP 404, and unofficial mirrors
were not used. VisA was chosen instead because it is official, directly downloadable, has PCB (electronics)
categories and a more permissive licence (CC BY 4.0).

The code is ready for MVTec AD: `mesq.vision.data.load_mvtec_style()` reads the standard MVTec folder layout.
If MVTec AD is obtained through the official form, remember it is **non-commercial (NC) and share-alike (SA)**.

## 3. Pretrained weights - official sources only

| Model | Source | Licence | Local cache |
|---|---|---|---|
| WideResNet-50-2, `Wide_ResNet50_2_Weights.IMAGENET1K_V1` | torchvision (download.pytorch.org) | torchvision BSD-3-Clause; weights trained on ImageNet-1k | `~/.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth` (138 MB, downloaded by this project) |
| BAAI/bge-small-en-v1.5 (sentence embeddings) | Hugging Face hub | MIT (model card) | already present in `~/.cache/huggingface/hub`; loaded with `HF_HUB_OFFLINE=1`, nothing downloaded |

No LLM is used. A local Ollama install exists on the development machine, but its server was not running when the
project was built, so per the project rules it was not used and no model was pulled.

## 4. Synthetic content written for this project

* `mesq/generator.py` - MES data (work orders, lots, genealogy, equipment, operators `OP-###`, suppliers
  `SUP-X`, material batches, ground-truth causes). Fully determined by `GenConfig` and the seed.
* `sop/docs/*.md` - nine SOP / work-instruction documents; each starts with a SYNTHETIC DOCUMENT banner. Values
  are plausible but fictional and must not be used for real production.
* `sop/questions.jsonl` - 42 questions (10 dev, 32 test) written before the retrieval code existed; its SHA-256 and
  freeze time are in `sop/questions.sha256`.
