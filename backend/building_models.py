"""Bina tespit modeli sağlayıcıları. Seçilen model çalışmazsa başka bir modele geçilmez."""
import hashlib
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from prediction_pipeline import PIPELINE_VERSION, deduplicate_predictions

ROOT = Path(__file__).resolve().parent
_LOCK = threading.Lock()


def weight_path(provider):
    name = 'GEOAI_WEIGHTS' if provider == 'geoai' else 'YOLO_WEIGHTS'
    default = ROOT / 'models' / ('building_footprints_usa.pth' if provider == 'geoai' else 'best.pt')
    return Path(os.getenv(name, str(default))).resolve()


@lru_cache(maxsize=8)
def _digest(path, size, mtime):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def model_info(provider):
    from vectorize import ROBOFLOW_MODEL_ID, ROBOFLOW_API_KEY
    if provider == 'roboflow':
        return {'provider': provider, 'model_id': ROBOFLOW_MODEL_ID,
                'ready': bool(ROBOFLOW_API_KEY), 'weights_sha256': None}
    if provider not in ('geoai', 'yolo'):
        raise ValueError('Unknown model provider')
    path = weight_path(provider)
    digest = _digest(str(path), path.stat().st_size, path.stat().st_mtime_ns) if path.is_file() else None
    return {'provider': provider, 'model_id': f'{provider}/{path.name}', 'ready': path.is_file(),
            'weights_sha256': digest, 'reason': None if path.is_file() else f'Missing weights: {path.name}'}


@lru_cache(maxsize=2)
def _load_local(provider, path, digest, device):
    if provider == 'yolo':
        from ultralytics import YOLO
        model = YOLO(path, task='segment')
        names = list(model.names.values()) if isinstance(model.names, dict) else model.names
        if len(names) != 1 or str(names[0]).lower() not in ('building', 'buildings', 'bina'):
            raise ValueError('YOLO must be a single-class building segmentation checkpoint, not COCO weights.')
        return model
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    # Mimari ve RGB normalizasyonu GeoAI ObjectDetector ile aynıdır.
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2,
        image_mean=[.485, .456, .406], image_std=[.229, .224, .225])
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'model' in state:
        state = state['model']
    elif 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    return model.to(device).eval()


def scan_buildings(image, *, provider='roboflow', confidence_threshold=.25,
                   tile_size=640, overlap=128, target_size=None, processing_profile='conservative'):
    from vectorize import scan_satellite_image_roboflow, _tile_starts
    if provider == 'roboflow':
        return scan_satellite_image_roboflow(image, confidence_threshold=confidence_threshold,
            tile_size=tile_size, overlap=overlap, target_size=target_size,
            processing_profile=processing_profile)
    if processing_profile == 'legacy':
        raise ValueError('Legacy processing is only supported for the Roboflow baseline.')
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError('Invalid tile dimensions')
    info = model_info(provider)
    if not info['ready']:
        raise ValueError(info['reason'])
    import torch
    image = image.convert('RGB')
    if target_size and image.size != target_size:
        raise ValueError('Local inference requires the original image size; resampling is not implicit.')
    device = os.getenv('MODEL_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    predictions, raw_tiles, timings = [], [], []
    # Eşzamanlı bellek taşmasını önlemek için GPU çağrıları ve model yükleme sıraya alınır.
    with _LOCK:
        model = _load_local(provider, str(weight_path(provider)), info['weights_sha256'], device)
        for y in _tile_starts(image.height, tile_size, overlap):
            for x in _tile_starts(image.width, tile_size, overlap):
                crop = [x, y, min(x + tile_size, image.width), min(y + tile_size, image.height)]
                tile = image.crop(crop)
                started = time.perf_counter()
                local = []
                if provider == 'geoai':
                    tensor = torch.from_numpy(np.asarray(tile).copy()).permute(2, 0, 1).float().div(255).to(device)
                    with torch.inference_mode():
                        # Çıkarım eşiği, istenen güven eşiğinin altında tutulur.
                        model.roi_heads.score_thresh = min(.05, confidence_threshold)
                        output = model([tensor])[0]
                    for mask, score, label in zip(output['masks'], output['scores'], output['labels']):
                        score = float(score)
                        if int(label) != 1 or score < confidence_threshold:
                            continue
                        binary = (mask[0].detach().cpu().numpy() >= .5).astype('uint8')
                        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if len(contour) >= 3 and cv2.contourArea(contour) > 0:
                                local.append((contour.reshape(-1, 2).tolist(), score))
                else:
                    output = model.predict(tile, conf=confidence_threshold, imgsz=tile_size,
                                           device=device, verbose=False, retina_masks=True)[0]
                    if output.masks is not None:
                        local = [(points.tolist(), float(score)) for points, score in zip(output.masks.xy, output.boxes.conf)]
                tile_predictions = []
                for points, score in local:
                    xs, ys = [p[0] for p in points], [p[1] for p in points]
                    pred = {'points': [{'x': float(px + x), 'y': float(py + y)} for px, py in points],
                            'confidence': score, 'class': 'building', 'offset_applied': True,
                            'tile': crop, 'prediction_source': provider,
                            'detection_id': f'{provider}_{x}_{y}_{len(tile_predictions)}',
                            'width': max(xs)-min(xs), 'height': max(ys)-min(ys),
                            'x': (min(xs)+max(xs))/2+x, 'y': (min(ys)+max(ys))/2+y,
                            '_cut': bool((min(xs) <= 3 and x > 0) or (min(ys) <= 3 and y > 0)
                                or (max(xs) >= tile.width-3 and crop[2] < image.width)
                                or (max(ys) >= tile.height-3 and crop[3] < image.height))}
                    tile_predictions.append(pred)
                predictions.extend(tile_predictions)
                raw_tiles.append({'crop_box': crop, 'predictions': tile_predictions})
                timings.append(round(time.perf_counter()-started, 3))
    return {**info, 'predictions': predictions if processing_profile == 'raw' else deduplicate_predictions(predictions),
            'raw_predictions': predictions, 'raw_tiles': raw_tiles,
            'image': {'width': image.width, 'height': image.height}, 'tiles': len(raw_tiles),
            'failed_tiles': 0, 'tile_timings_s': timings, 'device': device,
            'processing_profile': processing_profile, 'pipeline_version': PIPELINE_VERSION}
