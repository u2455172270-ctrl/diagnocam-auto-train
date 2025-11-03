import os, io, json, time, datetime
from pathlib import Path
from ultralytics import YOLO
from supabase import create_client

# === Config da variabili d'ambiente (impostate nei Secrets di GitHub Actions) ===
SUPABASE_URL         = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]  # service role key
BUCKET               = os.environ.get("BUCKET", "diagnocam")

EPOCHS = int(os.environ.get("EPOCHS", "120"))
IMGSZ  = int(os.environ.get("IMGSZ", "832"))
BATCH  = int(os.environ.get("BATCH", "8"))
MODEL_BASE = os.environ.get("MODEL_BASE", "yolov8s.pt")  # oppure path a un tuo best.pt

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def sb_list(prefix: str):
    return sb.storage.from_(BUCKET).list(prefix) or []

def sb_download_dir(prefix: str, dest: Path):
    ensure_dir(dest)
    entries = sb_list(prefix)
    c = 0
    for e in entries:
        name = e.get("name")
        if not name or name.endswith("/"):
            continue
        data = sb.storage.from_(BUCKET).download(f"{prefix.rstrip('/')}/{name}")
        (dest / name).write_bytes(data)
        c += 1
    return c

def sb_move(old_key: str, new_key: str):
    data = sb.storage.from_(BUCKET).download(old_key)
    sb.storage.from_(BUCKET).upload(new_key, io.BytesIO(data), {"upsert": True})
    sb.storage.from_(BUCKET).remove([old_key])

def make_empty_labels_for_negatives(img_dir: Path, lbl_dir: Path):
    ensure_dir(lbl_dir)
    c = 0
    for img in img_dir.iterdir():
        if img.suffix.lower() not in [".jpg",".jpeg",".png"]:
            continue
        txt = lbl_dir / f"{img.stem}.txt"
        if not txt.exists():
            txt.write_text("")  # nessuna carie
            c += 1
    return c

def main():
    t0 = time.time()
    base = Path("workspace")
    images_train = base/"images/train"
    labels_train = base/"labels/train"
    ensure_dir(images_train); ensure_dir(labels_train)

    print("⬇️ Scarico nuovi file da Supabase…")
    n_img = sb_download_dir("incoming/images", images_train)
    n_lbl = sb_download_dir("incoming/labels", labels_train)
    empty = make_empty_labels_for_negatives(images_train, labels_train)
    print(f"📦 Immagini: {n_img} | Label: {n_lbl} | Label vuoti creati: {empty}")

    data_yaml = base/"data.yaml"
    data_yaml.write_text(
        f"train: {images_train}\n"
        f"val: {images_train}\n"
        f"nc: 1\n"
        f"names: ['carie']\n"
    )

    print("🚀 Training YOLO…")
    model = YOLO(MODEL_BASE)
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project="runs",
        name="auto_train_actions",
        seed=0,
        deterministic=True,
        patience=60,
        close_mosaic=10,
        verbose=True
    )

    run_dir = Path(results.save_dir)
    best_pt = run_dir/"weights"/"best.pt"
    results_png = run_dir/"results.png"

    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    model_key = f"models/best_{today}.pt"
    sb.storage.from_(BUCKET).upload(model_key, best_pt.read_bytes(), {"upsert": True})
    if results_png.exists():
        sb.storage.from_(BUCKET).upload(f"models/results_{today}.png", results_png.read_bytes(), {"upsert": True})
    sb.storage.from_(BUCKET).upload(
        "models/latest.json",
        io.BytesIO(json.dumps({"path": model_key, "date": today}).encode()),
        {"content-type":"application/json", "upsert": True}
    )

    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for f in sb_list("incoming/images"):
        name = f.get("name")
        if name and not name.endswith("/"):
            sb_move(f"incoming/images/{name}", f"archive/{stamp}/images/{name}")
    for f in sb_list("incoming/labels"):
        name = f.get("name")
        if name and not name.endswith("/"):
            sb_move(f"incoming/labels/{name}", f"archive/{stamp}/labels/{name}")

    print(json.dumps({
        "status":"ok",
        "downloaded_images": n_img,
        "downloaded_labels": n_lbl,
        "empty_labels_created": empty,
        "best_model": model_key,
        "elapsed_min": round((time.time()-t0)/60,2)
    }, indent=2))

if __name__ == "__main__":
    main()
