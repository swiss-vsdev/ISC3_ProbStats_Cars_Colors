import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.transforms.functional import to_tensor

CAR_CLASS_ID = 3  # index de la classe "car" dans COCO
SCORE_MIN = 0.6   # confiance minimale pour accepter une détection
IMAGES_DIR = Path("/Volumes/Seagate/autoscout24_images/1998")
OUTPUT_CSV = Path("resultats.csv")
EXTENSIONS = {".jpg", ".jpeg", ".png"}
COLONNES = ["image", "voiture_detectee", "confiance", "x1", "y1", "x2", "y2", "couleur", "rgb"]

# Palette de couleurs nommées (RGB)
PALETTE = {
    "black":       (20, 20, 20),
    "white":       (245, 245, 245),
    "gray":        (128, 128, 128),
    "silver":      (192, 192, 192),
    "anthracite":  (60, 62, 66),
    "blue":        (30, 70, 190),
    "red":         (190, 30, 30),
    "bordeaux":    (110, 20, 40),
    "brown":       (110, 70, 40),
    "beige":       (210, 190, 150),
    "green":       (30, 130, 60),
    "yellow":      (240, 200, 30),
    "orange":      (240, 120, 20),
    "gold":        (190, 150, 60),
    "turquoise":   (40, 170, 170),
    "violet":      (120, 60, 150),
    "pink":        (240, 150, 190),
}

# Modèle pré-entraîné, chargé une seule fois
weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).eval()


def couleur_dominante(voiture: Image.Image) -> np.ndarray:
    """Couleur majoritaire de la carrosserie (zone centrale, sans roues ni fond)."""
    w, h = voiture.size
    zone = voiture.crop((int(w * 0.2), int(h * 0.35), int(w * 0.8), int(h * 0.75)))
    pixels = np.asarray(zone.resize((64, 64))).reshape(-1, 3).astype(float)
    km = KMeans(n_clusters=3, n_init=3, random_state=0).fit(pixels)
    cluster_majoritaire = np.bincount(km.labels_).argmax()
    return km.cluster_centers_[cluster_majoritaire]


def nom_couleur(rgb: np.ndarray) -> str:
    """Associe un RGB à la couleur nommée la plus proche."""
    return min(PALETTE, key=lambda nom: np.linalg.norm(rgb - np.array(PALETTE[nom])))


def analyser_image(chemin: Path) -> list[dict]:
    img = Image.open(chemin).convert("RGB")
    with torch.no_grad():
        sortie = model([to_tensor(img)])[0]

    voitures = []
    for box, label, score in zip(sortie["boxes"], sortie["labels"], sortie["scores"]):
        if label.item() == CAR_CLASS_ID and score.item() >= SCORE_MIN:
            x1, y1, x2, y2 = map(int, box.tolist())
            rgb = couleur_dominante(img.crop((x1, y1, x2, y2)))
            voitures.append({
                "image": chemin.name,
                "voiture_detectee": True,
                "confiance": round(score.item(), 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "couleur": nom_couleur(rgb),
                "rgb": "-".join(str(int(c)) for c in rgb),
            })

    if not voitures:
        voitures.append({"image": chemin.name, "voiture_detectee": False})
    return voitures


def images_valides(dossier: Path):
    """Parcourt récursivement le dossier et ignore les fichiers cachés macOS (._*)."""
    for chemin in sorted(dossier.rglob("*")):
        if (
            chemin.is_file()
            and not chemin.name.startswith(("._", "."))
            and chemin.suffix.lower() in EXTENSIONS
        ):
            yield chemin


def main():
    lignes = []
    for chemin in images_valides(IMAGES_DIR):
        try:
            lignes.extend(analyser_image(chemin))
        except Exception as e:
            print(f"Ignoré {chemin} : {e}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(lignes)

    print(f"{len(lignes)} lignes écrites dans {OUTPUT_CSV}")


if __name__ == "__main__":
    main()