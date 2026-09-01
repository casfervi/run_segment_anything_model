# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:16:34 2026

@author: vinicius.ferreira

git - https://github.com/facebookresearch/sam3

Correções feitas:
- salvamento efetivo dos resultados (máscaras + metadata) em output/<prompt>_<timestamp>/
- modo vídeo processa em blocos de frames (chunking), cada um em sua própria
  sessão, para evitar o vazamento de memória do predictor em vídeos longos
  (ver sam3#305)
- correção das chaves de saída do video predictor (out_binary_masks /
  out_boxes_xywh / out_probs), diferentes das do processor de imagem
  (masks / boxes / scores)
- vídeo anotado final inclui TODOS os frames (com e sem detecção), não só
  os que tiveram detecção
- suporte a múltiplos prompts (objetos) numa mesma execução, separados por
  vírgula, cada um com sua própria cor na anotação
- sessão de vídeo é fechada ao final ("close_session")
- checagem de argc corrigida
- tratamento de erros para path inexistente / falha de modelo
"""

import contextlib
import os
import shutil
import sys
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor


def resolve_device() -> str:
    """
    Escolhe 'cuda' ou 'cpu'. Além de checar se há GPU disponível, testa
    se a build do PyTorch instalada realmente suporta a capacidade CUDA
    dessa GPU (ex: GPUs antigas como Pascal/sm_61 não são suportadas por
    builds recentes do PyTorch, o que gera "no kernel image is available").
    Pode ser forçado via variável de ambiente SAM3_DEVICE=cpu|cuda.
    """
    forced = os.environ.get("SAM3_DEVICE")
    if forced:
        return forced

    if not torch.cuda.is_available():
        return "cpu"

    try:
        (torch.zeros(1, device="cuda") + 1).cpu()
        return "cuda"
    except RuntimeError as e:
        print(
            "Aviso: GPU detectada, mas incompatível com esta build do PyTorch "
            f"({e}). Usando CPU. Defina SAM3_DEVICE=cuda para forçar mesmo assim, "
            "ou reinstale um PyTorch compatível com a capacidade CUDA da sua GPU."
        )
        return "cpu"


def autocast_ctx(device: str):
    """
    Sam3Processor.set_image/set_text_prompt (e o video predictor) não
    envolvem a inferência num contexto de autocast internamente. Em GPU,
    isso causa "mat1 and mat2 must have the same dtype, but got BFloat16
    and Float", pois partes do checkpoint carregam em bfloat16. A correção
    oficial (ver sam3#526 e os notebooks de exemplo do repositório) é
    envolver manualmente as chamadas de inferência neste contexto.
    """
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _sanitize(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")


def _make_output_dir(base_output: Path, prompts: list) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "_".join(_sanitize(p) for p in prompts)
    out_dir = base_output / f"{safe_prompt}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _mask_to_uint8(mask, target_size=None) -> np.ndarray:
    """Converte uma máscara (tensor torch ou array numpy, booleana/float) para uint8 0-255.
    Se target_size=(width, height) for informado e o shape não bater, redimensiona."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.squeeze(mask)
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    if target_size is not None and (mask_u8.shape[1], mask_u8.shape[0]) != target_size:
        mask_u8 = np.array(Image.fromarray(mask_u8).resize(target_size, Image.NEAREST))
    return mask_u8


def _save_masks(masks, out_dir: Path, prefix: str = "mask", target_size=None):
    saved = []
    for i, mask in enumerate(masks):
        mask_img = Image.fromarray(_mask_to_uint8(mask, target_size))
        mask_path = out_dir / f"{prefix}_{i:03d}.png"
        mask_img.save(mask_path)
        saved.append(mask_path.name)
    return saved


def _to_list(x):
    """Converte tensors/arrays para listas simples serializáveis em JSON."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


_BOX_COLORS = [
    (255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 180, 0),
    (255, 0, 200), (0, 220, 220), (150, 0, 255), (255, 100, 0),
]


def _color_for_prompt(prompt: str, prompts: list):
    """Cada prompt (objeto) recebe sempre a mesma cor, baseada na posição
    dele na lista de prompts pedida pelo usuário."""
    idx = prompts.index(prompt) if prompt in prompts else 0
    return _BOX_COLORS[idx % len(_BOX_COLORS)]


def _draw_boxes(image: Image.Image, boxes, scores=None, labels=None, colors=None, width: int = 6) -> Image.Image:
    """Desenha os bounding boxes (formato xyxy, coordenadas em pixel) sobre a imagem.
    labels: texto por caixa (ex: nome do prompt). Se None, usa o índice.
    colors: cor por caixa. Se None, cicla a paleta padrão."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    boxes_list = _to_list(boxes) or []
    scores_list = _to_list(scores) if scores is not None else [None] * len(boxes_list)
    labels_list = labels if labels is not None else [str(i) for i in range(len(boxes_list))]

    for i, (box, score, label) in enumerate(zip(boxes_list, scores_list, labels_list)):
        x1, y1, x2, y2 = box
        color = colors[i] if colors is not None else _BOX_COLORS[i % len(_BOX_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = str(label) + (f" {score:.2f}" if score is not None else "")
        text_y = max(0, y1 - 14)
        draw.rectangle([x1, text_y, x1 + 8 * len(text) + 4, text_y + 12], fill=color)
        draw.text((x1 + 2, text_y), text, fill=(255, 255, 255))

    return annotated


def _boxes_xywh_norm_to_xyxy_px(boxes_xywh, width: int, height: int):
    """Converte boxes normalizados (0-1) no formato xywh (formato do video
    predictor: out_boxes_xywh) para pixels absolutos no formato xyxy."""
    result = []
    for box in boxes_xywh:
        x, y, w, h = box
        result.append([x * width, y * height, (x + w) * width, (y + h) * height])
    return result


def run_image(media_path: Path, prompts: list, out_dir: Path, processor=None, device=None):
    """Se processor/device não forem passados, carrega o modelo aqui (uso
    single-shot). Ao processar uma pasta inteira, main() carrega o modelo
    uma única vez e reaproveita entre as imagens via esses parâmetros."""
    owns_model = processor is None
    if owns_model:
        print(f"Iniciando processamento de imagem para prompts: {prompts}")
        device = resolve_device()
        print(f"Usando device: {device}")
        model = build_sam3_image_model(device=device)
        processor = Sam3Processor(model)

    image = Image.open(media_path).convert("RGB")

    all_boxes, all_scores, all_labels, all_colors = [], [], [], []
    mask_files = []
    n_by_prompt = {}

    with autocast_ctx(device):
        # A imagem só precisa ser codificada uma vez; cada prompt reaproveita
        # o mesmo inference_state (mais rápido que recarregar por prompt).
        inference_state = processor.set_image(image)
        for prompt in prompts:
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            n = len(masks) if masks is not None else 0
            n_by_prompt[prompt] = n
            print(f"  '{prompt}': {n} instância(s) encontrada(s)")
            if not n:
                continue

            safe = _sanitize(prompt)
            mask_files += _save_masks(masks, out_dir, prefix=f"mask_{safe}")

            boxes_list = _to_list(boxes)
            scores_list = _to_list(scores) if scores is not None else [None] * n
            color = _color_for_prompt(prompt, prompts)

            all_boxes += boxes_list
            all_scores += scores_list
            all_labels += [prompt] * n
            all_colors += [color] * n

    n_found = len(all_boxes)
    annotated_file = None
    if n_found:
        annotated = _draw_boxes(image, all_boxes, all_scores, labels=all_labels, colors=all_colors)
        annotated_path = out_dir / "annotated.png"
        annotated.save(annotated_path)
        annotated_file = annotated_path.name
        print(f"Imagem anotada salva em: {annotated_path}")

    metadata = {
        "type": "image",
        "source_file": str(media_path),
        "prompts": prompts,
        "timestamp": datetime.now().isoformat(),
        "n_detections_by_prompt": n_by_prompt,
        "n_detections_total": n_found,
        "boxes": all_boxes,
        "scores": all_scores,
        "labels": all_labels,
        "mask_files": mask_files,
        "annotated_file": annotated_file,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Resultados salvos em: {out_dir}")


def _find_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".tif", ".tiff", ".webp"}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _get_video_frame(video_path: Path, frame_idx: int):
    """Extrai um frame do vídeo como PIL.Image (requer opencv-python)."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _video_info(video_path: Path):
    """Retorna (total_frames, fps) do vídeo (requer opencv-python)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return total, fps


def _extract_chunk_frames(video_path: Path, start: int, end: int, chunk_dir: Path):
    """Extrai os frames [start, end) do vídeo como JPEGs numerados em chunk_dir."""
    import cv2
    chunk_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    local_idx = 0
    for _ in range(end - start):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(chunk_dir / f"{local_idx:06d}.jpg"), frame)
        local_idx += 1
    cap.release()
    return local_idx


def _process_video_chunk(video_predictor, frames_dir: Path, prompt: str, device: str):
    """Roda uma sessão completa (prompt + propagação) sobre uma pasta de frames JPEG
    para UM prompt. Cada prompt usa sua própria sessão (tentar múltiplos prompts na
    mesma sessão de vídeo é instável no SAM3 atual, ver sam3#206)."""
    with autocast_ctx(device):
        response = video_predictor.handle_request(
            request=dict(type="start_session", resource_path=str(frames_dir))
        )
        session_id = response["session_id"]
        try:
            video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=prompt,
                )
            )
            outputs_per_frame = {}
            for response in video_predictor.handle_stream_request(
                request=dict(type="propagate_in_video", session_id=session_id)
            ):
                outputs_per_frame[response["frame_index"]] = response["outputs"]
        finally:
            video_predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
    return outputs_per_frame


def _build_annotated_video(out_dir: Path, frames_meta: dict, fps: float, media_path: Path):
    """Monta o vídeo final com TODOS os frames: usa annotated.png nos frames
    com detecção e o frame original (sem alteração) nos demais."""
    import cv2

    cap = cv2.VideoCapture(str(media_path))
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None
    h, w = frame.shape[:2]

    video_path = out_dir / "annotated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    frame_idx = 0
    while ok:
        meta = frames_meta.get(frame_idx)
        if meta and meta.get("annotated_file"):
            annotated_path = out_dir / f"frame_{frame_idx:05d}" / meta["annotated_file"]
            annotated_img = cv2.imread(str(annotated_path))
            if annotated_img is not None:
                if (annotated_img.shape[1], annotated_img.shape[0]) != (w, h):
                    annotated_img = cv2.resize(annotated_img, (w, h))
                writer.write(annotated_img)
            else:
                writer.write(frame)
        else:
            writer.write(frame)

        frame_idx += 1
        ok, frame = cap.read()

    cap.release()
    writer.release()
    return video_path.name


def run_video(media_path: Path, prompts: list, out_dir: Path):
    print(f"Iniciando processamento de vídeo para prompts: {prompts}")
    device = resolve_device()
    print(f"Usando device: {device}")
    if device != "cuda":
        # O predictor de vídeo do SAM3 só suporta CUDA; não há caminho de
        # CPU/MPS aqui ainda (ver sam3#164).
        print(
            "Aviso: o modo vídeo do SAM3 requer GPU CUDA compatível "
            "(não há suporte a CPU no video predictor). A execução pode falhar."
        )

    try:
        import cv2  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "opencv-python é necessário para o modo vídeo (extração/anotação de "
            "frames). Instale com: pip install opencv-python"
        )

    # Vídeos longos fazem a memória da sessão do SAM3 crescer sem limite
    # durante a propagação (bug conhecido, ver sam3#305). Processamos em
    # blocos de frames, cada bloco em sua própria sessão, liberando o cache
    # da GPU entre eles.
    chunk_frames = int(os.environ.get("SAM3_CHUNK_FRAMES", "150"))

    total_frames, fps = _video_info(media_path)
    print(f"Vídeo com {total_frames} frames ({fps:.2f} fps). Processando em blocos de {chunk_frames} frames.")

    # Além do close_session, o predictor de vídeo do SAM3 acumula memória
    # "presa" entre sessões mesmo com empty_cache() (ver sam3#305) — depois
    # de muitas sessões seguidas isso já não é só falta de VRAM, o tracker
    # fica progressivamente mais lento (o frame_filter passa a varrer um
    # banco de memória inflado). Recriar o predictor do zero a cada N
    # sessões evita esse acúmulo. Ajustável via SAM3_REBUILD_EVERY.
    rebuild_every = int(os.environ.get("SAM3_REBUILD_EVERY", "5"))
    session_count = 0

    video_predictor = build_sam3_video_predictor()

    opencv_warned = False
    frames_meta = {}
    n_processed = 0
    n_by_prompt = {p: 0 for p in prompts}

    with tempfile.TemporaryDirectory(prefix="sam3_chunks_") as tmp_root:
        tmp_root = Path(tmp_root)
        n_chunks = (total_frames + chunk_frames - 1) // chunk_frames

        for chunk_i in range(n_chunks):
            start = chunk_i * chunk_frames
            end = min(start + chunk_frames, total_frames)
            print(f"Bloco {chunk_i + 1}/{n_chunks}: frames {start}-{end - 1}")

            chunk_dir = tmp_root / f"chunk_{chunk_i:04d}"
            n_extracted = _extract_chunk_frames(media_path, start, end, chunk_dir)
            if n_extracted == 0:
                shutil.rmtree(chunk_dir, ignore_errors=True)
                continue

            # Roda cada prompt separadamente sobre o mesmo bloco de frames
            # (frames já extraídos são reaproveitados entre prompts) e
            # combina as detecções por frame local.
            per_frame_detections = {}  # local_idx -> list de dicts
            for prompt in prompts:
                try:
                    outputs_per_frame = _process_video_chunk(video_predictor, chunk_dir, prompt, device)
                finally:
                    session_count += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if session_count % rebuild_every == 0:
                        print(f"Reiniciando o video predictor (sessão {session_count}) "
                              "para liberar memória acumulada...")
                        del video_predictor
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        video_predictor = build_sam3_video_predictor()

                for local_idx, outputs in outputs_per_frame.items():
                    masks = outputs.get("out_binary_masks")
                    boxes_xywh = outputs.get("out_boxes_xywh")
                    scores = outputs.get("out_probs")
                    if masks is None or len(masks) == 0:
                        continue
                    n_by_prompt[prompt] += len(masks)
                    per_frame_detections.setdefault(local_idx, []).append(
                        dict(prompt=prompt, masks=masks, boxes_xywh=boxes_xywh, scores=scores)
                    )

            for local_idx, det_list in per_frame_detections.items():
                global_idx = start + local_idx
                frame_dir = out_dir / f"frame_{global_idx:05d}"
                frame_dir.mkdir(exist_ok=True)

                # Reaproveita o JPEG já extraído para este bloco em vez de
                # abrir o vídeo de novo (evita um segundo decode/seek no
                # arquivo original, caro em HEVC com GOP problemático como
                # o desse vídeo).
                chunk_frame_path = chunk_dir / f"{local_idx:06d}.jpg"
                frame_img = Image.open(chunk_frame_path).convert("RGB") if chunk_frame_path.exists() else None
                target_size = frame_img.size if frame_img is not None else None

                mask_files, all_boxes, all_scores, all_labels, all_colors = [], [], [], [], []
                for det in det_list:
                    safe = _sanitize(det["prompt"])
                    mask_files += _save_masks(det["masks"], frame_dir, prefix=f"mask_{safe}", target_size=target_size)
                    boxes_xywh = det["boxes_xywh"]
                    scores = det["scores"]
                    n = len(det["masks"])
                    if frame_img is not None and boxes_xywh is not None:
                        boxes_xyxy = _boxes_xywh_norm_to_xyxy_px(_to_list(boxes_xywh), *frame_img.size)
                        all_boxes += boxes_xyxy
                        all_scores += _to_list(scores) if scores is not None else [None] * n
                        all_labels += [det["prompt"]] * n
                        all_colors += [_color_for_prompt(det["prompt"], prompts)] * n

                annotated_file = None
                if frame_img is not None and all_boxes:
                    annotated = _draw_boxes(frame_img, all_boxes, all_scores, labels=all_labels, colors=all_colors)
                    annotated_path = frame_dir / "annotated.png"
                    annotated.save(annotated_path)
                    annotated_file = annotated_path.name
                elif frame_img is None and not opencv_warned:
                    print("Aviso: falha ao ler frame original para anotação.")
                    opencv_warned = True

                frames_meta[global_idx] = {
                    "mask_files": mask_files,
                    "labels": all_labels,
                    "scores": all_scores,
                    "annotated_file": annotated_file,
                }
                n_processed += 1

            shutil.rmtree(chunk_dir, ignore_errors=True)

    print(f"{n_processed} frame(s) com detecção, de {total_frames} processados no total.")
    print(f"Detecções por prompt: {n_by_prompt}")

    annotated_video = None
    if frames_meta:
        print("Remontando vídeo completo (todos os frames, com e sem detecção)...")
        annotated_video = _build_annotated_video(out_dir, frames_meta, fps, media_path)
        if annotated_video:
            print(f"Vídeo anotado salvo em: {out_dir / annotated_video}")

    metadata = {
        "type": "video",
        "source_file": str(media_path),
        "prompts": prompts,
        "timestamp": datetime.now().isoformat(),
        "total_frames": total_frames,
        "fps": fps,
        "chunk_frames": chunk_frames,
        "n_frames_with_detection": n_processed,
        "n_detections_by_prompt": n_by_prompt,
        "annotated_video": annotated_video,
        "frames": frames_meta,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Resultados salvos em: {out_dir}")


def run_image_folder(folder: Path, prompts: list, out_dir: Path):
    """Processa todas as imagens de uma pasta, carregando o modelo uma
    única vez e reaproveitando entre elas. Cada imagem ganha sua própria
    subpasta dentro de out_dir; um summary.json consolida os totais."""
    image_files = _find_images(folder)
    if not image_files:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {folder}")

    print(f"{len(image_files)} imagem(ns) encontrada(s) em {folder}")
    device = resolve_device()
    print(f"Usando device: {device}")
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model)

    summary = {}
    for i, img_path in enumerate(image_files, 1):
        print(f"--- [{i}/{len(image_files)}] {img_path.name} ---")
        img_out_dir = out_dir / img_path.stem
        img_out_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary[img_path.name] = run_image(img_path, prompts, img_out_dir, processor=processor, device=device)
        except Exception as e:
            print(f"Erro ao processar {img_path.name}: {e}")
            summary[img_path.name] = f"erro: {e}"

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Resumo salvo em: {out_dir / 'summary.json'}")


def main(media_folder: str, video_or_image: str, prompt: str, output_folder: str = "output"):
    media_path = Path(media_folder)
    if not media_path.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {media_path}")

    # Múltiplos objetos: separados por vírgula. Ex: "rust,flange"
    prompts = [p.strip() for p in prompt.split(",") if p.strip()]
    if not prompts:
        raise ValueError("Informe ao menos um prompt (nome do objeto a detectar)")

    base_output = Path(output_folder)
    out_dir = _make_output_dir(base_output, prompts)

    if video_or_image == "image":
        if media_path.is_dir():
            run_image_folder(media_path, prompts, out_dir)
        else:
            run_image(media_path, prompts, out_dir)
    elif video_or_image == "video":
        run_video(media_path, prompts, out_dir)
    else:
        raise ValueError("Escolha 'video' ou 'image' para o segundo argumento")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: python run_sam3.py <media_folder> <video_or_image> "<prompt1>[,<prompt2>,...]" [output_folder]')
        sys.exit(1)

    media_folder_arg = sys.argv[1]
    video_or_image_arg = sys.argv[2]
    prompt_arg = sys.argv[3]
    output_folder_arg = sys.argv[4] if len(sys.argv) > 4 else "output"

    try:
        main(media_folder_arg, video_or_image_arg, prompt_arg, output_folder_arg)
    except Exception as e:
        print(f"Erro: {e}")
        sys.exit(1)