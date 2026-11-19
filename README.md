# run_sam3.py

> **This repository is NOT SAM3 itself.** It is just a convenience wrapper
> script to more easily run the **SAM3 (Segment Anything Model 3)** model,
> developed and released by **Meta AI / FAIR (Facebook AI Research)**. All
> credit for the model, weights, and underlying code belongs to Meta.

Further resources:
- https://huggingface.co/facebook/sam3
- https://github.com/facebookresearch/sam3
- https://ai.meta.com/research/sam3/

---

README in Portuguese - README em portugu�s:

[![pt-br](https://img.shields.io/badge/lang-pt--br-green.svg)](https://github.com/casfervi/run_segment_anything_model/blob/main/README_pt.md)


---

Script to run SAM3 on images or videos using one or more text prompts,
saving masks, an annotated image/video with bounding boxes, and metadata
in a timestamped output folder.

## Activating the environment

Before running the script, build and activate a conda environment.

## Usage

```bash
python run_sam3.py <media_path> <image|video> "<prompt1>[,<prompt2>,...]" [output_folder]
```

- `media_path`: path to an image (jpg, png, jfif, ...), an **image
  folder**, or a video (mp4, or a folder of JPEG frames)
- `image` or `video`: media type
- `prompt`: one or more concepts to segment, **comma-separated**
  (e.g. `"flange"`, `"rust,flange,pipe"`). Each object gets its own fixed
  color in the drawn boxes.
- `output_folder` (optional): base folder where results are saved.
  Default: `output`

### Examples

Segment a flange in an image:

```bash
conda activate conda_env
python run_sam3.py C:\path\image.jpg image "flange"
```

Segment rust and flange in the same image:

```bash
conda activate conda_env
python run_sam3.py C:\path\image.jpg image "rust,flange"
```

Process every image in a folder at once (the model is loaded once and
reused across all images):

```bash
conda activate conda_env
python run_sam3.py C:\path\image_folder image "rust,flange"
```

Segment corrosion in a video:

```bash
conda activate conda_env
python run_sam3.py C:\path\video.mp4 video "rust"
```

Multiple objects in the same video (runs one pass per prompt over each
frame chunk — more prompts means more total time):

```bash
python run_sam3.py C:\path\video.mp4 video "rust,flange,pipe"
```

Specifying a different output folder:

```bash
python run_sam3.py C:\path\image.jpg image "corrosion" C:\path\results
```

Forcing CPU usage (e.g. if the GPU isn't compatible with the installed
PyTorch build — only works in image mode, see notes below):

```bash
set SAM3_DEVICE=cpu
python run_sam3.py C:\path\image.jpg image "pipe"
```

## What gets saved

Each run creates a folder:

```
output/<prompt1>_<prompt2>_..._<YYYYMMDD_HHMMSS>/
```

**Image mode:**
- `mask_<prompt>_000.png`, `mask_<prompt>_001.png`, ... — one binary mask
  per detection, prefixed with the prompt name
- `annotated.png` — original image with bounding boxes for all prompts
  drawn (one color per prompt)
- `metadata.json` — prompts, boxes, scores, labels (which prompt produced
  each detection), `n_detections_by_prompt`, and the list of generated
  files

If `media_path` is a **folder**, the script processes every image in it
(jpg, jpeg, png, bmp, jfif, tif, tiff, webp), loading the model once and
reusing it across images. Each image gets its own subfolder
`<image_name>/` inside the output folder, and a `summary.json` is written
at the root with the detection count per image.

**Video mode:**
- a subfolder `frame_00000/`, `frame_00001/`, ... for every frame with a
  detection, each with its masks (prefixed by prompt) and `annotated.png`
- `annotated_video.mp4` — a video containing **every original frame**:
  frames with a detection show the drawn boxes, the rest are kept exactly
  as in the source video
- `metadata.json` — prompts, total frame count, fps, chunk size used,
  `n_detections_by_prompt`, and per-frame details

> Annotating video frames requires `opencv-python`
> (`pip install opencv-python`). Without it, video mode won't run.

## Long videos: chunked processing and environment variables

Long videos make SAM3's session accumulate memory without bound during
propagation (a known SAM3 bug — [issue #305](https://github.com/facebookresearch/sam3/issues/305)),
causing `CUDA out of memory` partway through the video even on GPUs with
plenty of VRAM. To work around this, the script:

1. **Splits the video into frame chunks** and processes each chunk in its
   own session, with `close_session` + `torch.cuda.empty_cache()` between
   them — also reusing the already-extracted chunk JPEG to build the
   annotated frame, instead of re-opening/decoding the source video again.
2. Since that alone isn't enough for very long videos — the predictor
   still accumulates "stuck" memory between sessions, making processing
   progressively slower until it stalls — the script also **rebuilds the
   `video_predictor` from scratch periodically**, every N sessions (each
   chunk × each prompt counts as one session).

Two environment variables control this behavior:

- **`SAM3_CHUNK_FRAMES`** (default: `250`) — how many frames to process
  per chunk. Lower it if you still run out of GPU memory within a single
  chunk. Chunks that are too small increase session-startup overhead
  (more frame/model loading calls).
  ```bash
  set SAM3_CHUNK_FRAMES=150
  ```

- **`SAM3_REBUILD_EVERY`** (default: `10`) — how many sessions between
  full rebuilds of the `video_predictor`. This makes the overall run a
  bit slower (reloading the model takes a few seconds each time), but
  prevents stalling from accumulated memory on long videos or with many
  prompts. If it still stalls/slows down before reaching the default
  interval, lower it:
  ```bash
  set SAM3_REBUILD_EVERY=5
  ```

Example combining both:

```bash
set SAM3_CHUNK_FRAMES=150
set SAM3_REBUILD_EVERY=5
python run_sam3.py C:\path\video.mp4 video "rust,flange"
```

## Requirements / notes

- **Folder structure**: `run_sam3.py` must be run from the root of the
  cloned SAM3 repository (https://github.com/facebookresearch/sam3), i.e.
  the `sam3\` folder (the model's Python package, e.g.
  `sam3\model_builder.py`) needs to sit in the same directory you run
  `python run_sam3.py ...` from — that's what makes
  `from sam3.model_builder import ...` work. If you'd rather run it from
  anywhere, install the package into the environment with
  `pip install -e .` from inside the cloned repository folder.
- **Checkpoint access**: `facebook/sam3` is a gated repository on Hugging
  Face. You need to request access at https://huggingface.co/facebook/sam3
  and authenticate (`huggingface-cli login` or the `HF_TOKEN` environment
  variable) before the first run. If using the environment variable, set
  it directly in the terminal:
  ```bash
  set HF_TOKEN=hf_lettersandnumbers
  ```
- **`triton` on Windows**: SAM3's video tracker depends on the `triton`
  package. On Windows, the official PyPI package doesn't work — use the
  Windows-specific build instead:
  ```bash
  pip install -U triton-windows
  ```
- **GPU without Flash Attention** (pre-Ampere GPUs, compute capability <
  8.0): the script already handles this automatically by wrapping
  inference in `torch.autocast(dtype=torch.bfloat16)`.
- **GPU incompatible with the installed PyTorch build**: the script
  detects this case (`no kernel image is available` error) and falls back
  to CPU automatically in image mode, with a warning in the terminal.
- **Video mode requires a CUDA GPU**: unlike image mode, SAM3's
  `video_predictor` has no CPU/MPS support yet (it's implemented
  internally with CUDA/NCCL — [issue #164](https://github.com/facebookresearch/sam3/issues/164)).
  `SAM3_DEVICE=cpu` does not work for video.
- **Multiple prompts in video mode**: each prompt runs in its own session
  over the same extracted frames (mixing multiple prompts in the same
  video session is unstable in current SAM3 —
  [issue #206](https://github.com/facebookresearch/sam3/issues/206)), so
  total runtime scales with the number of requested objects.

## Credits

This script is only a convenience layer (chunking, multiple prompts,
image/video annotation, compatibility fixes) for running the **SAM3**
model. The model itself, its weights, and the underlying library were
created by **Meta AI / FAIR**:
- Model and weights: https://huggingface.co/facebook/sam3
- Official source code: https://github.com/facebookresearch/sam3
- Research page: https://ai.meta.com/research/sam3/
