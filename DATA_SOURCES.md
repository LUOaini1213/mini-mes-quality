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
| Expected location | `data/visa/raw/{pcb1,pcb4,split_csv,LICENSE-DATASET}` inside the repository (about 515 MB; `data/` is git-ignored). Override with `MESQ_VISA_ROOT`. |

Only `pcb1`, `pcb4`, `split_csv` and `LICENSE-DATASET` need to be extracted. VisA is published only as one archive
(no per-category files), so the whole 1.9 GB archive has to be downloaded once; the SHA-256 above lets you verify
the download, and the archive can be deleted after extraction.

Citation:

> Y. Zou, J. Jeong, L. Pemula, D. Zhang, O. Dabeer. *SPot-the-Difference Self-Supervised Pre-training for Anomaly
> Detection and Segmentation.* ECCV 2022. arXiv:2207.14315.

Attribution (CC BY 4.0): images (c) the VisA authors / Amazon, used unmodified for evaluation; this repository
redistributes no images. Scores and file names in `results/vision/*.csv` are derived data.

## 2. MVTec AD - licence checked, not downloaded

MVTec AD was the first choice. Its official page (https://www.mvtec.com/company/research/datasets/mvtec-ad,
checked 2026-10-13) states, verbatim:

> The data is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
> (CC BY-NC-SA 4.0). In particular, it is not allowed to use the dataset for commercial purposes. If you are unsure
> whether or not your application violates the non-commercial use clause of the license, please contact us.

Why it was not used: the page now delivers the dataset only after a registration web form ("Fill in the form")
that asks for personal details, the per-category archive links that used to be public
(`www.mydrive.ch/shares/38536/...`) return HTTP 404, unofficial mirrors were not used, and the licence is
non-commercial and share-alike. VisA was chosen instead because it is official, directly downloadable, has PCB
(electronics) categories and a more permissive licence (CC BY 4.0).

The code is ready for MVTec AD: `mesq.vision.data.load_mvtec_style()` reads the standard MVTec folder layout.
If MVTec AD is obtained through the official form, remember it is **non-commercial (NC) and share-alike (SA)**.

## 3. Pretrained weights - official sources only

| Model | Source | Licence | Local cache |
|---|---|---|---|
| WideResNet-50-2, `Wide_ResNet50_2_Weights.IMAGENET1K_V1` | torchvision (download.pytorch.org) | torchvision BSD-3-Clause; weights trained on ImageNet-1k | `~/.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth` (138 MB, downloaded by this project) |
| BAAI/bge-small-en-v1.5 (sentence embeddings) | Hugging Face hub | MIT (model card) | `~/.cache/huggingface/hub` after the one-time download in README "Reproduce" (about 130 MB); run time loads it with `HF_HUB_OFFLINE=1` |

No LLM is used: answers are extractive by design (the best-matching SOP clause, verbatim, with its citation), so
every answer is supported by the documents word for word.

## 4. Synthetic content written for this project

* `mesq/generator.py` - MES data (work orders, lots, genealogy, equipment, operators `OP-###`, suppliers
  `SUP-X`, material batches, ground-truth causes). Fully determined by `GenConfig` and the seed.
* `sop/docs/*.md` - nine SOP / work-instruction documents; each starts with a SYNTHETIC DOCUMENT banner. Values
  are plausible but fictional and must not be used for real production.
* `sop/questions.jsonl` - 42 questions (10 dev, 32 test); its SHA-256 and freeze time are in
  `sop/questions.sha256`. That the file was frozen before the retrieval code was written is the author's own
  record; the public git history is a single commit and cannot show the order.
