"""YOLOv8-seg modelini yalnızca denetlenmiş, elle etiketlenmiş veri setiyle eğitir."""
import argparse
import hashlib
import json
from pathlib import Path


def train(dataset, epochs=50, device=None, export_onnx=False):
    manifest_path = dataset.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('labels') != 'complete_independent_manual' or manifest.get('split_policy') != 'disjoint_geographic_windows':
        raise ValueError('Use prepare_training_data.py to export a reviewed dataset first.')
    if epochs < 1:
        raise ValueError('Positive epoch count required')
    from ultralytics import YOLO
    model = YOLO('yolov8n-seg.pt')
    model.train(data=str(dataset.resolve()),epochs=epochs,imgsz=manifest['tile_size'],
                seed=2026,deterministic=True,workers=0,device=device,
                project=str(Path(__file__).parent/'runs'),name='buildings',exist_ok=False)
    best = Path(model.trainer.best)
    provenance = {'dataset_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  'initial_weights':'yolov8n-seg.pt','seed':2026,'epochs':epochs,
                  'best_weights':str(best.resolve()),'test_evaluated':False}
    if export_onnx:
        provenance['onnx_path']=str(YOLO(str(best)).export(format='onnx',imgsz=manifest['tile_size'],simplify=False))
        provenance['onnx_parity_verified']=False
    best.with_suffix('.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--device',default=None)
    parser.add_argument('--export-onnx',action='store_true')
    args=parser.parse_args()
    train(args.data,args.epochs,args.device,args.export_onnx)
