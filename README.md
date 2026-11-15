# altitude-models

The model files of [Altitude](https://github.com/papito/altitude). They are published as the assets of tagged
releases, not committed: four are over git's 100 MB limit, and Git LFS bandwidth is metered. This repository holds the
script that makes them and this page.

A release's file is at:

    https://github.com/renegade-otter/altitude-models/releases/download/<tag>/<file>

## The files

| File | Size | What it is | Source | Licence |
|---|---|---|---|---|
| `dinov2_vitb14.onnx` | 346 MB | DINOv2 ViT-B/14, the pooled CLS token (768 floats): a Scene model | [`facebook/dinov2-base`](https://huggingface.co/facebook/dinov2-base), exported to ONNX | [Apache 2.0](LICENSE-APACHE-2.0) |
| `siglip2_b16_256_image.onnx` | 372 MB | SigLIP 2 ViT-B/16 image tower at 256 px, its pooled output (768 floats): a Scene model | [`google/siglip2-base-patch16-256`](https://huggingface.co/google/siglip2-base-patch16-256), exported to ONNX | [Apache 2.0](LICENSE-APACHE-2.0) |
| `siglip2_b16_256_text.onnx` | 343 MB | The same checkpoint's text tower, without its token-embedding table: a description's embedding | `google/siglip2-base-patch16-256`, exported to ONNX | [Apache 2.0](LICENSE-APACHE-2.0) |
| `siglip2_token_embeddings.f16` | 393 MB | That table: 256,000 rows of 768 little-endian float16, row = token ID | `google/siglip2-base-patch16-256` | [Apache 2.0](LICENSE-APACHE-2.0) |
| `siglip2_tokenizer.tsv` | 6 MB | The checkpoint's tokenizer, a line per piece: ID, merge score, matched whole, piece | `google/siglip2-base-patch16-256` | [Apache 2.0](LICENSE-APACHE-2.0) |
| `face_detection_yunet_2023mar.onnx` | 233 kB | YuNet, the face detector: face boxes and five landmarks | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet) at commit `f12e127`, as published | [MIT](LICENSE-YUNET-MIT), Copyright (c) 2020 Shiqi Yu |
| `SHA256SUMS` | | The SHA-256 of each file above, in the format of `sha256sum` | | |
| `LICENSE-APACHE-2.0`, `LICENSE-YUNET-MIT`, `NOTICE` | | The licences of the files above and a notice of their sources, as in this repository | | |

The four SigLIP 2 files are one checkpoint split for the way Altitude reads it: the table stays out of the text tower's
graph so that it can be mapped into memory, where a description reads 64 of its rows. `export-models.py` describes each
file's inputs, outputs and format, and the checks an export passes against PyTorch before it is written.

To check a downloaded set:

    sha256sum -c SHA256SUMS

## Licences

The DINOv2 and SigLIP 2 files are derived from their checkpoints, which are under the Apache License 2.0: the text is
in [`LICENSE-APACHE-2.0`](LICENSE-APACHE-2.0). YuNet is under the MIT License, in
[`LICENSE-YUNET-MIT`](LICENSE-YUNET-MIT), the licence file of YuNet's directory in OpenCV Zoo at the pinned commit.
[`NOTICE`](NOTICE) lists each file's source and what was done to it: the DINOv2 and SigLIP 2 files are converted to
ONNX from the original checkpoints, and SigLIP 2's text side is split into a text tower, a float16 token-embedding
table and a tokenizer table. The three files are attached to every release beside the model files.

## Releases

The checkpoint revisions each release is exported from, and the versions of torch and transformers it is exported
with. YuNet is the file of OpenCV Zoo at commit `f12e12798e8314f7c074a6656816c048dcc95b7a` in every release.

| Tag | `facebook/dinov2-base` | `google/siglip2-base-patch16-256` | torch | transformers |
|---|---|---|---|---|
| `v1` | not recorded | not recorded | 2.15.0 | not recorded |

The files of v1 were exported before the checkpoint revisions were pinned, DINOv2's on 2026-11-04 and SigLIP 2's on
2026-11-14, so v1's exact source revisions are not recorded; its torch version is the one its ONNX files name as their
producer. Later releases record theirs: the revisions are the ones `export-models.py` pins (`DINOV2_REVISION`,
`SIGLIP2_REVISION`), and the versions the ones its run prints.

## How Altitude uses a release

Altitude pins one release tag and, for every file, its URL, its size and its SHA-256, in code. It downloads the files it
is missing once, into its models directory, and checks each against its pin. The pins in the application are what is
trusted; `SHA256SUMS` is for people.

A release's model files are never changed once the application pins them. A new set of files, even one re-exported
from the same checkpoints (an export is not byte for byte the same across versions of torch), is a new tag, and the
application moves to it with new pins.

## Making a release

    rm -rf dist
    uv run export-models.py
    gh release create vN --title vN --notes "..." \
        dist/dinov2_vitb14.onnx \
        dist/siglip2_b16_256_image.onnx \
        dist/siglip2_b16_256_text.onnx \
        dist/siglip2_token_embeddings.f16 \
        dist/siglip2_tokenizer.tsv \
        dist/face_detection_yunet_2023mar.onnx \
        dist/SHA256SUMS \
        LICENSE-APACHE-2.0 LICENSE-YUNET-MIT NOTICE

`export-models.py` downloads YuNet from its pinned commit and checks it against its pinned SHA-256, exports the Scene
models and SigLIP 2's text side from their Hugging Face checkpoints, checking each export against PyTorch, and writes
everything with `SHA256SUMS` into `dist/`, which git ignores. Its dependencies are pinned in its header; `uv` installs
them, the CPU build of torch among them. `uv run export-models.py --help` lists its options.

A release starts from an empty `dist/`. The script skips a file already there by its name alone, so that a run that
failed part of the way picks up where it stopped; a file left from an earlier release would be shipped again. The
upload names its files, the ones the script lists in `RELEASE_FILES` with `SHA256SUMS`, and the licences and `NOTICE`
from the repository. `SHA256SUMS` covers the release files alone, and the script fails without writing it if `dist/`
holds anything else, such as the `.part` or `.unchecked` file a killed run leaves.

A new release adds its row to the table under [Releases](#releases), with the revisions `export-models.py` pins and the
versions of torch and transformers its run prints.

## Why ArcFace is not here

Altitude's face embedding net is ArcFace, InsightFace's `w600k_r50`. Its licence is InsightFace's non-commercial
research licence, so it is not re-hosted here: the application downloads it from the pinned Hugging Face revision it is
tuned with, and checks it against its pinned SHA-256.
