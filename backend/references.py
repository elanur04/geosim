"""Model çıktılarından ve düzeltilmiş haritalardan ayrı tutulan, sürümlü elle çizilmiş referanslar."""
import json
import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import analysis_archive
from analysis_archive import load_manifest
from evaluate_buildings import evaluate_reference
from regions import validate_collection

DB_PATH = Path(__file__).resolve().parent / 'data' / 'references.sqlite3'
router = APIRouter(prefix='/api/references', tags=['Independent manual reference'])


class ReferenceSave(BaseModel):
    revision: int = Field(ge=0)
    collection: dict


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=10)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS reference_versions (run_id TEXT, revision INTEGER, payload TEXT, PRIMARY KEY(run_id,revision))')
        with db:
            yield db
    finally:
        db.close()


def _load(db, run_id):
    row = db.execute('SELECT payload FROM reference_versions WHERE run_id=? ORDER BY revision DESC LIMIT 1', (run_id,)).fetchone()
    return json.loads(row[0]) if row else None


def _reference_key(manifest):
    """Aynı görüntüdeki farklı eşik denemeleri tek bir referansı paylaşır."""
    seen = set()
    while manifest.get('locked_inputs_from_run_id'):
        if manifest['id'] in seen:
            raise HTTPException(409, 'Döngüsel analiz kaydı bağlantısı.')
        seen.add(manifest['id'])
        parent = load_manifest(manifest['locked_inputs_from_run_id'])
        if (parent['bounds'] != manifest['bounds'] or parent['artifacts']['image.png']['sha256'] !=
                manifest['artifacts']['image.png']['sha256']):
            raise HTTPException(409, 'Eşik karşılaştırmasının görüntüleri aynı değil.')
        manifest = parent
    return manifest['id']


@router.get('/{run_id}')
def get_reference(run_id: str):
    manifest = load_manifest(run_id)
    with _connect() as db:
        result = _load(db, _reference_key(manifest))
    if result is None:
        raise HTTPException(404, 'Bu görüntü için henüz elle işaretlenmiş referans yok.')
    return result


@router.put('/{run_id}')
def save_reference(run_id: str, req: ReferenceSave):
    manifest = load_manifest(run_id)
    if manifest['status'] != 'complete':
        raise HTTPException(422, 'Kısmi analiz doğruluk değerlendirmesinde kullanılamaz.')
    data = validate_collection(req.collection)
    key = _reference_key(manifest)
    meta = data['meta']
    if (meta.get('role') != 'reference' or meta.get('annotation_method') != 'manual_from_image'
            or meta.get('bounds') != manifest['bounds']
            or meta.get('image_sha256') != manifest['artifacts']['image.png']['sha256']):
        raise HTTPException(422, 'Referans tam olarak seçilen görüntü ve sınırlar için çizilmelidir.')
    if any(f['properties'].get('source') != 'manual' or f['properties']['type'] != 'building' for f in data['features']):
        raise HTTPException(422, 'Referans katmanına yalnızca görüntüden elle çizilmiş binalar alınabilir.')
    data['meta']['analysis_run_id'] = key
    if meta.get('reference_complete') and not data['features']:
        raise HTTPException(422, 'Boş referans tamamlandı olarak işaretlenemez.')
    try:
        for field in ('second_reviewer_fraction', 'second_reviewer_area_fraction'):
            fraction = float(meta.get(field, 0))
            if not 0 <= fraction <= 1:
                raise ValueError()
            if fraction and (not str(meta.get('second_reviewer', '')).strip() or not meta.get('reviewed_at')):
                raise ValueError()
        if meta.get('dataset_role', 'unassigned') not in ('unassigned', 'train', 'development', 'test'):
            raise ValueError()
        json.dumps(data, allow_nan=False)
    except (TypeError, ValueError):
        raise HTTPException(422, 'Geçersiz referans veya denetim oranı.')
    with _connect() as db:
        db.execute('BEGIN IMMEDIATE')
        old = _load(db, key)
        revision = old['revision'] if old else 0
        if revision != req.revision:
            raise HTTPException(409, 'Referans başka oturumda değişti; yeniden açın.')
        result = {'analysis_run_id': key, 'revision': revision + 1, 'collection': data,
                  'updated_at': datetime.now(timezone.utc).isoformat()}
        db.execute('INSERT INTO reference_versions VALUES (?,?,?)', (key, revision + 1, json.dumps(result, ensure_ascii=False)))
    return result


@router.post('/{run_id}/evaluate')
def evaluate_saved_reference(run_id: str):
    saved = get_reference(run_id)
    try:
        report = evaluate_reference(run_id, saved['collection'])
    except ValueError as error:
        raise HTTPException(422, str(error))
    report['reference_revision'] = saved['revision']
    report['advisor_review_requirement_met'] = (report['second_reviewer_fraction'] >= .2
        and report['second_reviewer_area_fraction'] >= .2
        and bool(report['second_reviewer']) and bool(report['reviewed_at']))
    report['measured_at'] = datetime.now(timezone.utc).isoformat()
    return report


@router.post('/{run_id}/compare-thresholds')
def compare_thresholds(run_id: str):
    reference = get_reference(run_id)
    if reference['collection']['meta'].get('dataset_role') != 'development':
        raise HTTPException(422, 'Eşik seçimi yalnızca D1 geliştirme referansında yapılabilir; test alanında seçim yapılamaz.')
    baseline_id = _reference_key(load_manifest(run_id))
    baseline = load_manifest(baseline_id)
    def osm_digest(manifest):
        path = analysis_archive.ROOT / manifest['id'] / 'osm.geojson'
        features = json.loads(path.read_text(encoding='utf-8'))['features']
        return hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest()
    baseline_osm_digest = osm_digest(baseline)
    latest = {}
    for path in analysis_archive.ROOT.glob('*/manifest.json'):
        try:
            candidate = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        if candidate['status'] != 'complete' or candidate['model_id'] != baseline['model_id']:
            continue
        if any(candidate.get(key) != baseline.get(key) for key in
               ('model_provider', 'weights_sha256', 'processing_profile', 'pipeline_version', 'code_sha256')):
            continue
        if candidate['id'] != baseline_id and candidate.get('locked_inputs_from_run_id') != baseline_id:
            continue
        candidate = load_manifest(candidate['id'])
        if (osm_digest(candidate) != baseline_osm_digest or candidate['tile_size'] != baseline['tile_size']
                or candidate['tile_overlap'] != baseline['tile_overlap']):
            raise HTTPException(422, 'Eşik koşularında aynı OSM verisi ve aynı karo düzeni kullanılmalı.')
        confidence = candidate['request']['confidence']
        if confidence not in (.25, .4, .6):
            continue
        if confidence not in latest or candidate['captured_at'] > latest[confidence]['captured_at']:
            latest[confidence] = candidate
    if set(latest) != {.25, .4, .6}:
        raise HTTPException(422, 'Aynı görüntüde 0.25, 0.40 ve 0.60 eşikleriyle üç tam model koşusu gerekli.')
    results = [evaluate_saved_reference(latest[value]['id']) for value in sorted(latest)]
    best_f1 = max(r['methods']['ai']['f1'] for r in results)
    return {'baseline_run_id': baseline_id, 'results': results,
            'selection_metric': 'ai_f1', 'selection_role': 'development',
            'best_thresholds_for_this_image': [r['inference_confidence'] for r in results if abs(r['methods']['ai']['f1'] - best_f1) < 1e-10],
            'selected_deployment_threshold': None,
            'note': 'D1 üzerinde AI F1 karşılaştırmasıdır. Seçimi nihai testten önce sabitleyin; test sonuçlarına göre eşik ayarlamayın.'}
