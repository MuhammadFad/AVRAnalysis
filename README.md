# PixelDelta — Perceptual Image Regression Testing for Game Renders

A parallelized computer vision pipeline for automated visual quality assurance of game rendering outputs. Compares a **baseline** (high quality) render against an **optimized** (lower quality) render to detect and quantify perceptual quality regression.

---

## 📦 Dataset

This project does not ship with a dataset. You have two options:

### Option A — GISET (quick start)
Download the GISET dataset (Gaming Image quality SET):
- Source: https://drive.google.com/open?id=1R2MDH6aNmhZwXFwdHmM91kGIxwR3krf5
- Place reference frames in `data/baseline/`
- Place encoded frames in `data/optimized/`

### Option B — UE5 Generated
Generate your own pairs from Unreal Engine 5:

1. Open any UE5 project with rich geometry and lighting
2. Place a `CineCameraActor` at a visually interesting position
3. Capture a **baseline** render via console:
```
sg.ShadowQuality 3
sg.TextureQuality 3
sg.PostProcessQuality 3
r.Lumen.Reflections.Allow 1
HighResShot 1
```
4. Capture the **optimized** render:
```
sg.ShadowQuality 0
sg.TextureQuality 0
sg.PostProcessQuality 0
r.Lumen.Reflections.Allow 0
HighResShot 1
```
5. Place output PNGs in the respective `baseline/` and `optimized/` folders

Filenames must match between the two folders (e.g. `shot_01.png` in both).

---

## ⚙️ Setup

**Requirements:** Python 3.10+

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```