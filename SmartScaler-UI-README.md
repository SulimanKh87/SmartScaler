# SmartScaler Web UI — One‑Click User Guide

This guide shows non‑technical users how to launch and use **SmartScaler Web UI** to batch‑process raster images:
1) validate & smart‑resize inputs to **256×256** (skips invalid images with a clear reason), and
2) upscale valid images to **512×512** with the pretrained model.

---

## 🚀 Quick Start (Windows 10/11)

**You only need to double‑click. No terminal commands.**

1. **Download / clone** the SmartScaler folder to your computer.
2. Double‑click **`run_smartscaler.bat`** at the project root.
   - First run: it **creates virtual environments** and **installs all requirements** automatically.
   - It then opens your browser at **http://localhost:8501** with the UI.
3. Use the web UI to upload images and start processing (details below).

> If Windows asks for permission to open a browser or run the script, click **Allow** / **Run anyway**.

### Optional (macOS users)
If your package includes it, double‑click **`run_smartscaler.command`**. The behavior is the same: it sets up a venv and launches the UI at http://localhost:8501.

---

## ✅ What the UI Does

- **Input**: one or more raster images (PNG, JPG, JPEG; others may be skipped).
- **Step 1 — Smart Resizer (to 256×256)**  
  Calls `data_processing/smart_resizer.py`.  
  - If an image **cannot** be made 256×256 under the project’s rules, it is **skipped** and shown with a **clear error message** in the UI.
  - Valid images are saved to the **256×256** folder for the next step.
- **Step 2 — Upscale to 512×512**  
  Uses the pretrained model (e.g., `saved_models/upscaler.h5`) to upscale the 256×256 results to **512×512**.  
  The UI displays large previews and a compact, per‑image result row (with quality labels when available).
- **Extras**
  - **Batch mode** (multiple files).
  - **Per‑image delete buttons** to remove items from the output folders.
  - **Large previews** of both 256×256 and 512×512 results.
  - **Quality score/label** (Good / Average / Bad) when enabled by your build.

---

## 📁 Where Files Are Saved

The launcher creates a working area under `ui_runtime/` inside your SmartScaler folder.

```
ui_runtime/
├── resizer_out/
│   ├── input_raster_256x256/    # step 1 outputs (valid images)
│   └── input_raster_512x512/    # (optional) if your resizer emits paired 512s
└── upscale_out/
    └── 512x512/                 # step 2 outputs (final upscaled images)
```

- **Skipped images** never appear in these folders; they will be listed in the **Errors** table in the UI with a reason.
- You can **delete** any output image from inside the UI (per‑image delete button).

---

## 🖱️ Using the Web UI (Step‑by‑Step)

1. **Open the UI**  
   After double‑clicking `run_smartscaler.bat`, your browser opens at **http://localhost:8501**.
2. **Upload Images**  
   - Drag & drop or use **Browse** to select multiple PNG/JPG/JPEG files.
3. **Run Smart Resizer (256×256)**  
   - Click **Run Resizer**.  
   - The UI shows:
     - **Processed count** and **Skipped count**.
     - A **Results** grid: each item in its own square with filename, size, and a thumbnail.
     - An **Errors** panel (if any) with per‑file reasons from `smart_resizer.py`.
4. **Run Upscale (512×512)**  
   - Click **Run Upscale** (or **Upscale + Complete**) to process the valid 256×256 images.  
   - You’ll see **big previews** of the 512×512 results and a compact one‑line summary per image (filename, metrics/labels).
5. **Manage Outputs (optional)**  
   - Use **Delete** beside any preview to remove that file from disk.
   - You can also open the folders shown above to copy the results wherever you need.

> **Tip:** You can re‑run steps on new batches at any time; only new files will be processed.

---

## 🔧 One‑Click Launcher Details (for the curious)

- `run_smartscaler.bat` is **automatic**:
  - Creates/updates three virtual environments:  
    - `ui_venv` — runs the Streamlit UI.  
    - `tf_venv` — runs TensorFlow‑based scripts (`smart_resizer.py`, upscaler).  
    - `venv310` — optional tooling/aux scripts.
  - Installs **all requirements up front**:  
    `requirementsUI.txt`, `requirementsTF_VENV.txt`, `requirementsVENV310.txt`.
  - Exposes these helpers for the UI to call the “right” Python without user input:
    - `SMARTSCALER_UI_PY` → `<project>/ui_venv/Scripts/python.exe`
    - `SMARTSCALER_TF_PY` → `<project>/tf_venv/Scripts/python.exe`
    - `SMARTSCALER_V310_PY` → `<project>/venv310/Scripts/python.exe`
  - The UI always runs in **`ui_venv`**; heavy jobs are spawned in **`tf_venv`** automatically. No manual switching.

You don’t need to interact with any of this — it’s here so the UI can “switch venv on click” under the hood.

---

## 🧪 Supported Files & Rules (Resizer)

- Typical formats: **PNG**, **JPG**, **JPEG**.
- Images that **cannot** be conformed to **256×256** according to `smart_resizer.py` logic are **skipped**.  
  The UI shows a reason from the script, e.g.:
  - “Unsupported file type”
  - “Image is corrupted or unreadable”
  - “Cannot scale to 256×256 under current constraints”
  - “Too small / zero dimension”

---

## 🧰 Troubleshooting

### “❌ CuPy not available — falling back to CPU”
That’s fine. The app runs on CPU by default. Performance will be slower than a CUDA build, but results are identical.

### “Resizer error: smart_resizer.py returned non‑zero exit status”
- Check the **Errors** table for per‑file messages.
- Make sure your inputs are valid images and not empty/locked.
- If this appears with **every** file, verify that the model/requirements were installed on first run (re‑launch the `.bat` to auto‑repair).

### The browser doesn’t open
- Visit **http://localhost:8501** manually after you double‑click the `.bat`.
- If nothing appears, your antivirus may have blocked Python/Streamlit — allow it and try again.

### “streamlit is not installed”
Re‑run `run_smartscaler.bat` so it can reinstall `requirementsUI.txt` into `ui_venv` automatically.

### Where are the logs?
- The terminal window that opened with the `.bat` shows a live log.
- Errors for individual files also appear inside the UI’s **Errors** panel.

---

## 🔒 Privacy & Local Processing

- All processing is **local** on your machine.
- Your images and results stay inside the **SmartScaler** folder unless you copy them elsewhere.

---

## ❓ FAQ

**Q: Can I run another batch after finishing one?**  
A: Yes. Upload more images and click **Run Resizer** again, then **Run Upscale**.

**Q: Can I remove a wrong result?**  
A: Yes. Use the **Delete** button beside that image in the UI, or delete it from the output folder.

**Q: Do I need a GPU?**  
A: No. The app runs on CPU by default. A supported GPU build can speed things up but isn’t required.

**Q: Which model is used for upscaling?**  
A: The UI loads the pretrained model packaged with your build (for example, `saved_models/upscaler.h5`). If it’s missing, the UI will show a friendly error.

---

## 🧹 Reset / Repair (optional, advanced)

If something looks broken (missing packages, etc.):  
1) Close the UI and any Python windows.  
2) Double‑click `run_smartscaler.bat` again — it will **re‑install** missing packages automatically.  
3) If you want a hard reset, you can delete the `ui_venv`, `tf_venv`, and `venv310` folders; the next launch will recreate them.

---

Happy scaling! 🎉
