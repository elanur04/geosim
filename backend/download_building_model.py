"""GeoAI RGB model ağırlığını sürüm ve SHA256 bilgisiyle indirir."""
import hashlib
import json
from pathlib import Path
import requests


def download():
    root = Path(__file__).resolve().parent / 'models'
    root.mkdir(exist_ok=True)
    metadata = requests.get('https://huggingface.co/api/models/giswqs/geoai', timeout=30)
    metadata.raise_for_status()
    revision = metadata.json()['sha']
    name = 'building_footprints_usa.pth'
    url = f'https://huggingface.co/giswqs/geoai/resolve/{revision}/{name}'
    path = root / name
    if path.exists():
        raise SystemExit('Weights already exist; refusing to overwrite. Use the saved manifest.')
    temp = path.with_suffix('.download')
    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        with temp.open('wb') as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
    temp.replace(path)
    manifest = {'provider': 'geoai', 'repo_id': 'giswqs/geoai', 'revision': revision,
                'filename': name, 'url': url, 'sha256': digest.hexdigest(),
                'license': 'MIT (model repository)', 'input': 'RGB',
                'architecture': 'torchvision maskrcnn_resnet50_fpn, num_classes=2',
                'local_accuracy_measured': False}
    path.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    download()
