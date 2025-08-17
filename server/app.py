from flask import Flask, request, jsonify, send_from_directory
import subprocess, sys, os, csv, shutil, threading, time, io, zipfile
from pathlib import Path
from werkzeug.utils import secure_filename

BASE = Path(__file__).parent.parent
UI_DIR = BASE/"ui"; RUNTIME = BASE/"ui_runtime"
UPLOADS = RUNTIME/"uploads"; RES_OUT = RUNTIME/"resizer_out"; COPIES = RUNTIME/"copies"
for p in (UPLOADS, RES_OUT, COPIES): p.mkdir(parents=True, exist_ok=True)

ALLOWED = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
RESIZER_PY = [Path("data_preprocessing")/"smart_resizer.py", Path("smart_resizer.py")]
BATCH_PY = [Path("data_processing")/"batch_inference_with_compare_metrics.py", Path("batch_inference_with_compare_metrics.py")]

PROGRESS = {"status":"idle","total":0,"done":0,"start_ts":0.0,"eta_sec":None,"sec_per_img":None,"message":""}
PROG_LOCK = threading.Lock()

def _tf_py():
    p = os.environ.get("SMARTSCALER_TF_PY","").strip()
    return p if p and Path(p).exists() else sys.executable

def _env():
    e = os.environ.copy()
    e["PYTHONUTF8"]="1"; e["PYTHONIOENCODING"]="utf-8"
    e.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION","python")
    e.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
    e.setdefault("TF_FORCE_GPU_ALLOW_GROWTH","true")
    return e

def _find(cands):
    for c in cands:
        if c.exists(): return c
    return None

def _read_csv(p: Path):
    if not p.exists(): return []
    import csv
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

app = Flask(__name__, static_folder=str(UI_DIR), static_url_path="")

@app.get("/")
def index(): return app.send_static_file("index.html")

@app.get("/<path:f>")
def static_files(f): return app.send_static_file(f)

@app.post("/api/upload")
def upload():
    if "files" not in request.files: return jsonify(ok=False, message="No files part"), 400
    saved=[]
    for f in request.files.getlist("files"):
        name = secure_filename(f.filename or "")
        if not name or Path(name).suffix.lower() not in ALLOWED: continue
        dst = UPLOADS/name; f.save(dst); saved.append(name)
    return jsonify(ok=True, saved=saved, upload_dir=str(UPLOADS))

@app.post("/api/run-resizer")
def run_resizer():
    s = _find(RESIZER_PY)
    if not s: return jsonify(ok=False, message="smart_resizer.py not found"), 404
    if RES_OUT.exists(): shutil.rmtree(RES_OUT, ignore_errors=True)
    RES_OUT.mkdir(parents=True, exist_ok=True)
    cmd = [_tf_py(), str(s), "--input_dir", str(UPLOADS), "--output_dir", str(RES_OUT)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_env())
    (COPIES/"resizer_outputs").mkdir(parents=True, exist_ok=True)
    for p in RES_OUT.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED:
            shutil.copy2(p, COPIES/"resizer_outputs"/p.name)
    for name in ("processed_images.csv","skipped_images.csv"):
        src = RES_OUT/name
        if src.exists(): shutil.copy2(src, COPIES/src.name)
    return jsonify(ok=(proc.returncode==0), stdout=proc.stdout, stderr=proc.stderr,
                   processed=_read_csv(RES_OUT/"processed_images.csv"),
                   skipped=_read_csv(RES_OUT/"skipped_images.csv"))

# Helpers for batch step
def _strip_suffix(stem: str):
    if stem.endswith("_256"): return stem[:-4], 256
    if stem.endswith("_512"): return stem[:-4], 512
    return stem, None

def _select_pairs(src256: Path, src512: Path, limit: int):
    map256 = {}
    for p in src256.glob("*"):
        if p.is_file() and p.suffix.lower() in [".png",".jpg",".jpeg"]:
            base, tag = _strip_suffix(p.stem); map256[base]=p
    map512 = {}
    for p in src512.glob("*"):
        if p.is_file() and p.suffix.lower() in [".png",".jpg",".jpeg"]:
            base, tag = _strip_suffix(p.stem); map512[base]=p
    bases = sorted(set(map256.keys()) & set(map512.keys()))
    if limit and limit < len(bases): bases = bases[:limit]
    return [(b, map256[b], map512[b]) for b in bases]

def _normalize_copy(src: Path, dst_dir: Path, base: str, tag: int):
    out = dst_dir / f"{base}_{tag}{src.suffix.lower()}"
    out.write_bytes(src.read_bytes()); return out

def _batch_thread(batch_script: Path, limit: int, source: str):
    project_root = batch_script.parent.parent if batch_script.parent.name=="data_processing" else batch_script.parent
    ds256 = project_root/"output/dataset/256/test/original"
    ds512 = project_root/"output/dataset/512/test/original"
    out = project_root/"output/test_output"
    for d in (ds256, ds512, out): d.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*"):
        if p.is_file(): p.unlink()

    if source == "uploads":
        # Use user inputs that passed resizer → copy into dataset layout
        src256 = RES_OUT/"input_raster_256x256"
        src512 = RES_OUT/"input_raster_512x512"
        pairs = _select_pairs(src256, src512, limit)
        total = len(pairs)
        with PROG_LOCK: PROGRESS.update({"status":"preparing","total":total,"done":0,"start_ts":time.time(),"eta_sec":None,"sec_per_img":None,"message":"copying from uploads"})
        for base, p256, p512 in pairs:
            _normalize_copy(p256, ds256, base, 256)
            _normalize_copy(p512, ds512, base, 512)
    else:
        # Use existing dataset; respect limit by pruning 256 set
        all256 = sorted([p for p in ds256.glob("*_256.*") if p.is_file()])
        if limit and limit < len(all256): all256 = all256[:limit]
        with PROG_LOCK: PROGRESS.update({"status":"preparing","total":len(all256),"done":0,"start_ts":time.time(),"eta_sec":None,"sec_per_img":None,"message":"using existing dataset"})

    # Monitor progress via appearance of *_upscaled.*
    def monitor():
        while True:
            time.sleep(1.0)
            done = sum(1 for p in out.glob("*_upscaled.*"))
            with PROG_LOCK:
                if PROGRESS["status"] not in ("running","preparing"): break
                PROGRESS["done"] = min(done, PROGRESS["total"])
                elapsed = max(time.time() - PROGRESS["start_ts"], 1e-6)
                if PROGRESS["done"] > 0:
                    sec_per = elapsed / PROGRESS["done"]
                    remaining = PROGRESS["total"] - PROGRESS["done"]
                    PROGRESS["sec_per_img"] = sec_per
                    PROGRESS["eta_sec"] = int(sec_per * remaining)
            if done >= PROGRESS["total"]:
                break
    mon = threading.Thread(target=monitor, daemon=True); mon.start()

    with PROG_LOCK: PROGRESS["status"]="running"; PROGRESS["message"]="upscaling & comparing"
    proc = subprocess.run([_tf_py(), str(batch_script)], cwd=str(project_root), capture_output=True, text=True, env=_env())

    csvp = out/"batch_compare_metrics.csv"
    (COPIES/"batch_outputs").mkdir(parents=True, exist_ok=True)
    if csvp.exists(): shutil.copy2(csvp, COPIES/"batch_outputs"/csvp.name)
    for p in out.glob("*"):
        if p.is_file() and p.suffix.lower() in [".png",".jpg",".jpeg",".webp"]:
            shutil.copy2(p, (COPIES/"batch_outputs"/p.name))

    with PROG_LOCK:
        PROGRESS.update({"status":"done","done":PROGRESS["total"],"eta_sec":0,"message":"finished","stdout":proc.stdout,"stderr":proc.stderr})

@app.post("/api/run-batch-upscale")
def run_batch():
    s = _find(BATCH_PY)
    if not s: return jsonify(ok=False, message="batch_inference_with_compare_metrics.py not found"), 404

    payload = request.get_json(force=True, silent=True) or {}
    limit = int(payload.get("limit", 0))
    source = payload.get("source","uploads")  # "uploads" or "dataset"

    if source == "uploads":
        src256 = RES_OUT/"input_raster_256x256"; src512 = RES_OUT/"input_raster_512x512"
        if not src256.exists() or not any(src256.iterdir()): return jsonify(ok=False, message="No 256 inputs from resizer"), 400
        if not src512.exists() or not any(src512.iterdir()): return jsonify(ok=False, message="No 512 targets from resizer"), 400

    with PROG_LOCK:
        PROGRESS.update({"status":"initializing","total":0,"done":0,"start_ts":time.time(),"eta_sec":None,"sec_per_img":None,"message":"initializing"})
    t = threading.Thread(target=_batch_thread, args=(s, limit, source), daemon=True); t.start()
    return jsonify(ok=True, message="started", source=source)

@app.get("/api/batch-progress")
def batch_progress():
    with PROG_LOCK:
        data = dict(PROGRESS)
    s = _find(BATCH_PY)
    if s:
        project_root = s.parent.parent if s.parent.name=="data_processing" else s.parent
        csvp = project_root/"output/test_output/batch_compare_metrics.csv"
        data["metrics"] = _read_csv(csvp)
    return jsonify(data)

@app.get("/api/list-copies")
def list_copies():
    imgs=[]
    for sub in ("resizer_outputs","batch_outputs"):
        d = COPIES/sub
        if d.exists():
            imgs += [f"{sub}/{p.name}" for p in d.glob("*") if p.suffix.lower() in [".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"]]
    csvs=[p.name for p in COPIES.glob("*.csv")]
    csvs += [f"batch_outputs/{p.name}" for p in (COPIES/'batch_outputs').glob('*.csv')] if (COPIES/'batch_outputs').exists() else []
    return jsonify(images=imgs, csvs=csvs)

@app.get("/copies/<path:rel>")
def copies(rel):
    f = (COPIES/rel); return send_from_directory(f.parent, f.name, as_attachment=("csv" in f.suffix))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5502, debug=False)
