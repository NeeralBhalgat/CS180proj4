import base64
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT / "images"
DATA_FILE = IMAGE_DIR / "poster_correspondences.json"
OUTPUT_FILE = IMAGE_DIR / "poster_correspondences.svg"

with DATA_FILE.open() as file:
    data = json.load(file)

points1 = data["im1Points"]
points2 = data["im2Points"]
if len(points1) != len(points2):
    raise ValueError("The two images need the same number of correspondence points.")

image_width = 480
image_height = 640
scale = image_width / 1152
left_x = 20
right_x = 500
image_y = 55

def image_data_uri(path):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


parts = [
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 715" '
    'role="img" aria-labelledby="title description">',
    '<title id="title">Poster photo correspondences</title>',
    '<desc id="description">The two poster photos side by side with colored '
    'lines connecting each manually matched feature.</desc>',
    '<rect width="1000" height="715" fill="#f6f5f1"/>',
    '<g font-family="Arial, sans-serif" fill="#222">',
    '<text x="20" y="34" font-size="20">Image 1</text>',
    '<text x="500" y="34" font-size="20">Image 2</text>',
    '<text x="615" y="34" font-size="13" fill="#62666b">Same-color lines connect matching points</text>',
    f'<image href="{image_data_uri(IMAGE_DIR / "posters_1.jpg")}" '
    f'x="{left_x}" y="{image_y}" '
    f'width="{image_width}" height="{image_height}"/>',
    f'<image href="{image_data_uri(IMAGE_DIR / "posters_2.jpg")}" '
    f'x="{right_x}" y="{image_y}" '
    f'width="{image_width}" height="{image_height}"/>',
]

for index, (point1, point2) in enumerate(zip(points1, points2)):
    x1, y1 = point1
    x2, y2 = point2
    start_x = left_x + x1 * scale
    end_x = right_x + x2 * scale
    start_y = image_y + y1 * scale
    end_y = image_y + y2 * scale
    color = f"hsl({index * 137.508 % 360:.1f}, 90%, 42%)"
    parts.append(
        f'<line x1="{start_x:.1f}" y1="{start_y:.1f}" '
        f'x2="{end_x:.1f}" y2="{end_y:.1f}" stroke="{color}" '
        'stroke-width="2" stroke-opacity="0.8"/>'
    )

for image_x, points in ((left_x, points1), (right_x, points2)):
    for index, (x, y) in enumerate(points):
        marker_x = image_x + x * scale
        marker_y = image_y + y * scale
        color = f"hsl({index * 137.508 % 360:.1f}, 90%, 42%)"
        parts.append(
            f'<circle cx="{marker_x:.1f}" cy="{marker_y:.1f}" r="4.5" '
            f'fill="{color}" stroke="white" stroke-width="1.5"/>'
        )

parts.append("</g></svg>")
OUTPUT_FILE.write_text("\n".join(parts))
print(f"Saved {OUTPUT_FILE}")
