import os, sys, csv, time, json, subprocess, argparse
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = Path(__file__).resolve().parent

# Paths
UI_RUNTIME = BASE_DIR / "ui_runtime"
UPLOADS = UI_RUNTIME / "uploads"
RESIZER_OUT = UI_RUNTIME / "resizer_out"
DIR256 = RESIZER_OUT / "input_raster_256x256"
DIR512 = RESIZER_OUT / "input_raster_512x512"
OUTPUT_DIR = BASE_DIR / "output" / "test_output"
ZIPS_DIR = BASE_DIR / "zips"

for p in [UPLOADS, DIR256, DIR512, OUTPUT_DIR, ZIPS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def find_tf_python() -> str | None:
    # 1) Env override
    env_path = os.environ.get('SMARTSCALER_TF_PY', '').strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return str(p)
    # 2) config.json override {"tf_python": "path\\to\\python.exe"}
    cfg = BASE_DIR / 'config.json'
    if cfg.exists():
        try:
            import json as _json
            j = _json.loads(cfg.read_text(encoding='utf-8'))
            tfp = j.get('tf_python')
            if tfp and Path(tfp).exists():
                return str(Path(tfp))
        except Exception:
            pass
    # 3) Common local venv names
    candidates = [
        BASE_DIR / 'tf_venv' / 'Scripts' / 'python.exe',
        BASE_DIR / 'TF_VENV' / 'Scripts' / 'python.exe',
        BASE_DIR / 'venv_tf' / 'Scripts' / 'python.exe',
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None

def smart_join_text(txt: str) -> str:
    # Windows terminals sometimes break on unicode; strip astral/emojis
    safe = "".join(ch for ch in txt if (ord(ch) < 0xD800 or ord(ch) > 0xDFFF) and ord(ch) <= 0x10FFFF)
    return safe

def run_cmd(args):
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(BASE_DIR), check=False)
        out = p.stdout.decode("utf-8", errors="replace")
        return p.returncode, smart_join_text(out)
    except Exception as e:
        return 1, f"ERROR: {e}"

def path_candidates(*names):
    for n in names:
        p = BASE_DIR / n
        if p.exists():
            return p
    return None

# Flask app
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")

@app.route("/")
def root():
    return app.send_static_file("index.html")

@app.post("/api/upload")
def api_upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify(ok=False, message="No files provided"), 400
    saved = 0
    for f in files:
        name = f.filename
        if not name:
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg"]:
            continue
        dest = UPLOADS / name
        f.save(dest)
        saved += 1
    return jsonify(ok=True, message=f"Saved {saved} file(s) to {UPLOADS}")

@app.post("/api/run-resizer")
def api_run_resizer():
    py = find_tf_python()
    script = path_candidates("data_preprocessing/smart_resizer.py", "smart_resizer.py")
    if script is None:
        return jsonify(ok=False, logs="smart_resizer.py not found."), 500

    logs = []
    cmd = [
        py, str(script),
        "--input_dir", str(UPLOADS),
        "--output_dir", str(RESIZER_OUT),
        "--batch_size", "8"
    ]
    rc, out = run_cmd(cmd)
    logs.append(out)
    return jsonify(ok=(rc==0), logs="\n".join(logs))

@app.get("/api/list-copies")
def api_list_copies():
    items = []
    # Pair images by stem removing suffix like _256/_512
    if not DIR256.exists():
        return jsonify(items=[])
    for p256 in sorted(DIR256.glob("*.*")):
        name = p256.name
        stem = name.replace("_256", "")
        p512 = DIR512 / name.replace("_256","_512")
        if not p512.exists():
            # try same stem if file naming didn't change
            alt = DIR512 / p256.name
            p512 = alt if alt.exists() else None
        if p512 and p512.exists():
            rel256 = str(p256.relative_to(UI_RUNTIME)).replace("\\","/")
            rel512 = str(p512.relative_to(UI_RUNTIME)).replace("\\","/")
            items.append({
                "name": stem,
                "rel256": rel256,
                "rel512": rel512,
                "url256": f"/copies/{rel256}",
                "url512": f"/copies/{rel512}"
            })
    return jsonify(items=items)

@app.get("/copies/<path:subpath>")
def copies(subpath):
    # Serve from ui_runtime
    d = UI_RUNTIME
    return send_from_directory(str(d), subpath, as_attachment=False)

@app.post("/api/run-upscale-compare")
def api_run_upscale():
    body = request.get_json(force=True, silent=True) or {}
    mode = body.get("mode","dataset")
    limit = int(body.get("limit", 0))

    py = find_tf_python()
    script = path_candidates("data_processing/batch_inference_with_compare_metrics.py",
                             "batch_inference_with_compare_metrics.py")
    if script is None:
        return jsonify(ok=False, logs="batch_inference_with_compare_metrics.py not found."), 500
    if not py:
        return jsonify(ok=False, logs=(
            "TensorFlow Python not found. Create a tf_venv next to this UI, or set either:\n"
            " - Environment var SMARTSCALER_TF_PY to the path of your TF python.exe\n"
            " - config.json with {\"tf_python\": \"C:/.../python.exe\"}\n"
        )), 500

    cmd = [py, str(script)]
    if mode == "uploads":
        cmd += ["--mode","uploads",
                "--inputs256", str(DIR256),
                "--targets512", str(DIR512)]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]

    rc, out = run_cmd(cmd)
    return jsonify(ok=(rc==0), logs=out)

@app.get("/api/get-metrics")
def api_get_metrics():
    csv_path = OUTPUT_DIR / "batch_compare_metrics.csv"
    rows = []
    if csv_path.exists():
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    # support both header orders
                    rows.append({
                        "filename": r.get("Filename", ""),
                        "psnr": r.get("PSNR", ""),
                        "ssim": r.get("SSIM", ""),
                        "mse": r.get("MSE", r.get("MSE ", ""))  # tolerate accidental space
                    })
        except Exception as e:
            rows = [{"filename":"(error)", "psnr":"", "ssim":"", "mse":str(e)}]
    return jsonify(rows=rows)

@app.post("/api/zip-upscale-outputs")
def api_zip_outputs():
    from zipfile import ZipFile, ZIP_DEFLATED
    name = f"test_output_{int(time.time())}.zip"
    zpath = ZIPS_DIR / name
    with ZipFile(zpath, "w", ZIP_DEFLATED) as z:
        if OUTPUT_DIR.exists():
            for p in OUTPUT_DIR.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(OUTPUT_DIR))
    url = f"/zips/{name}"
    return jsonify(url=url)

@app.get("/zips/<path:sub>")
def zips(sub):
    return send_from_directory(str(ZIPS_DIR), sub, as_attachment=True)


@app.get("/api/list-uploads")
def api_list_uploads():
    items = []
    if UPLOADS.exists():
        for p in sorted(UPLOADS.iterdir()):
            if p.is_file():
                items.append({"name": p.name, "path": str(p)})
    return jsonify(items=items)

@app.post("/api/delete-upload")
def api_delete_upload():
    body = request.get_json(force=True, silent=True) or {}
    name = body.get("name","")
    if not name:
        return jsonify(ok=False, error="name required"), 400
    target = (UPLOADS / name).resolve()
    if not target.exists() or not target.is_file():
        return jsonify(ok=False, error="file not found"), 404
    # Ensure inside uploads
    if UPLOADS.resolve() not in target.parents:
        return jsonify(ok=False, error="invalid path"), 400
    try:
        target.unlink()
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

@app.post("/api/delete-resizer-pair")
def api_delete_resizer_pair():
    body = request.get_json(force=True, silent=True) or {}
    rel256 = body.get("rel256","")
    rel512 = body.get("rel512","")

    # Only allow deletions inside ui_runtime/resizer_out
    def safe_del(rel):
        if not rel: return None, "missing path"
        p = (UI_RUNTIME / rel).resolve()
        if not p.exists(): return None, "not found"
        if (RESIZER_OUT.resolve() not in p.parents):
            return None, "invalid path"
        try:
            p.unlink()
            return True, ""
        except Exception as e:
            return None, str(e)

    ok256, e256 = safe_del(rel256)
    ok512, e512 = safe_del(rel512)

    if (ok256 or rel256=="") and (ok512 or rel512==""):
        return jsonify(ok=True)
    else:
        return jsonify(ok=False, error=f"256:{e256} 512:{e512}"), 400
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=5502, type=int)
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
