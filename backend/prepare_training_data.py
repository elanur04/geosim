"""Bağımsız etiketlenmiş, coğrafi olarak ayrık pencereleri YOLO-seg formatına aktarır.

Plan JSON: {"windows": [{"run_id": "...", "split": "train|val|test"}]}.
Eğitim/doğrulama/test ayrımı model seçiminden önce yapılır. AI/OSM çıktıları
etiketlere kopyalanmaz. Var olan çıktı klasörlerinin üzerine yazılmaz.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from PIL import Image
from shapely.geometry import box, shape
from shapely.ops import transform
from analysis_archive import load_manifest
import analysis_archive
from references import get_reference
from vectorize import _tile_starts


def prepare(plan, output, tile_size=640, overlap=128):
    if tile_size < 32 or not 0 <= overlap < tile_size:
        raise ValueError('Invalid tile dimensions')
    windows = plan.get('windows', [])
    if not windows or not {'train','val'} <= {w['split'] for w in windows}:
        raise ValueError('At least separate train and val windows with complete manual references are required.')
    loaded = []
    for window in windows:
        split = window['split']
        if split not in ('train','val','test'):
            raise ValueError('Split must be train, val or test')
        manifest = load_manifest(window['run_id'])
        if manifest['status'] != 'complete':
            raise ValueError('Partial image archive cannot be used for training')
        reference = get_reference(window['run_id'])
        data = reference['collection']
        role = {'train':'train', 'val':'development', 'test':'test'}[split]
        if data['meta'].get('dataset_role') != role or data['meta'].get('reference_complete') is not True:
            raise ValueError(f'{window["run_id"]}: complete manual reference with role {role} required')
        if any(f['properties'].get('source') != 'manual' for f in data['features']):
            raise ValueError('Only independent manual labels are permitted')
        bounds = manifest['bounds']
        extent = box(bounds['west'], bounds['south'], bounds['east'], bounds['north'])
        for other_split, other_extent, other_manifest, _ in loaded:
            if manifest['id'] == other_manifest['id']:
                raise ValueError('Repeated source run')
            if other_split != split and (extent.intersects(other_extent) or manifest['artifacts']['image.png']['sha256'] == other_manifest['artifacts']['image.png']['sha256']):
                raise ValueError('Geographic/image leakage between dataset splits')
        for feature in data['features']:
            geometry = shape(feature['geometry'])
            polys = [geometry] if geometry.geom_type=='Polygon' else list(geometry.geoms)
            if not geometry.is_valid or geometry.is_empty or not extent.covers(geometry):
                raise ValueError('Invalid/out-of-frame reference geometry')
            if any(p.interiors for p in polys):
                raise ValueError('YOLO polygon labels cannot represent courtyard holes. Use an instance-mask training pipeline; do not silently fill them.')
        loaded.append((split, extent, manifest, reference))
    if output.exists():
        raise ValueError('Output already exists; choose a new dataset version')
    records = []
    for split, _, manifest, reference in loaded:
        image = Image.open(analysis_archive.ROOT/manifest['id']/'image.png').convert('RGB')
        b = manifest['bounds']
        yn, ys = [math.asinh(math.tan(math.radians(b[k]))) for k in ('north','south')]
        def to_pixel(x,y,z=None):
            return ((x-b['west'])/(b['east']-b['west'])*image.width,
                    (yn-math.asinh(math.tan(math.radians(y))))/(yn-ys)*image.height)
        geometries = [transform(to_pixel, shape(f['geometry'])) for f in reference['collection']['features']]
        for folder in ('images','labels'):
            (output/folder/split).mkdir(parents=True, exist_ok=True)
        count = 0
        for y in _tile_starts(image.height,tile_size,overlap):
            for x in _tile_starts(image.width,tile_size,overlap):
                right,bottom = min(x+tile_size,image.width),min(y+tile_size,image.height)
                frame = box(x,y,right,bottom)
                lines = []
                for geom in geometries:
                    clipped = geom.intersection(frame)
                    if clipped.is_empty:
                        continue
                    parts = [clipped] if clipped.geom_type=='Polygon' else list(getattr(clipped,'geoms',[]))
                    for part in parts:
                        if part.geom_type!='Polygon' or part.area<=0:
                            continue
                        coords = [(max(0,min(1,(px-x)/(right-x))),max(0,min(1,(py-y)/(bottom-y)))) for px,py in part.exterior.coords[:-1]]
                        lines.append('0 '+' '.join(f'{value:.8f}' for p in coords for value in p))
                name = f'{manifest["id"]}_{x}_{y}'
                image.crop((x,y,right,bottom)).save(output/'images'/split/f'{name}.png')
                (output/'labels'/split/f'{name}.txt').write_text('\n'.join(lines),encoding='utf-8')
                count += 1
        raw_ref = json.dumps(reference['collection'],sort_keys=True).encode()
        records.append({'run_id':manifest['id'],'split':split,'image_sha256':manifest['artifacts']['image.png']['sha256'],
                        'reference_revision':reference['revision'],'reference_sha256':hashlib.sha256(raw_ref).hexdigest(),
                        'tiles':count, 'bounds':manifest['bounds']})
    dataset = {'path':str(output.resolve()),'train':'images/train','val':'images/val','names':{0:'building'}}
    if any(r['split']=='test' for r in records):
        dataset['test']='images/test'
    # JSON geçerli bir YAML olduğu için ek kütüphane gerekmez.
    (output/'dataset.yaml').write_text(json.dumps(dataset,indent=2),encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps({'windows':records,'tile_size':tile_size,'overlap':overlap,
        'split_policy':'disjoint_geographic_windows','labels':'complete_independent_manual'},indent=2),encoding='utf-8')
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(json.loads(args.plan.read_text(encoding='utf-8')),args.output),indent=2))
