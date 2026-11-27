from pathlib import Path
from PIL import Image
import csv


IMAGES_DIR = Path("/Volumes/Seagate/autoscout24_images/1998")
OUTPUT_CSV = Path("resultats_simple.csv")
COLONNES = ["image", "annee", "couleur", "rgb"]

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


def get_most_present_color(image_path:Path):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    zone = (width // 4, height // 4, width * 3 // 4, height * 3 // 4)
    img = img.crop(zone)

    color = dominant_color(img)
    c_color = closest_color(color)
    y = get_year(image_path)
    print("Dominant color in the middle 50%:", c_color)
    return f"{image_path};{y};{c_color};{color}"


def get_year(image_path: Path):
    annee = image_path.name[:4]
    return annee

def dominant_color(img:Image):
    colors = img.getcolors(maxcolors=img.width * img.height)
    count, rgb = max(colors)
    return rgb


def closest_color(rgb:tuple[int, int, int], palette:dict = PALETTE) -> str:
    r, g, b = rgb
    return min(
        palette,
        key=lambda name: (palette[name][0] - r) ** 2
                       + (palette[name][1] - g) ** 2
                       + (palette[name][2] - b) ** 2,
    )


def main():
    lines = []
    for image in IMAGES_DIR.iterdir():
        if image.is_file():
            try:
                lines.append(get_most_present_color(image))
            except:
                pass
    
    csv.writer(open(OUTPUT_CSV, "w", newline="")).writerow(COLONNES)

    for l in lines:
        csv.writer(open(OUTPUT_CSV, "a", newline="")).writerow(l.split(";"))


if __name__ == "__main__":
    main()