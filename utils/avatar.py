"""Developer portrait: uses assets/developer.jpg|png if present, else generates an initials avatar."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def avatar_path() -> str:
    for name in ("developer.jpg", "developer.jpeg", "developer.png"):
        p = ROOT / "assets" / name
        if p.exists():
            return str(p)
    from PIL import Image, ImageDraw, ImageFont
    out = ROOT / "assets" / "developer_avatar.png"
    if not out.exists():
        out.parent.mkdir(exist_ok=True)
        S = 400
        img = Image.new("RGB", (S, S))
        px = img.load()
        for y in range(S):
            for x in range(S):
                t = (x + y) / (2 * S)
                px[x, y] = (int(99 + t * (6 - 99)), int(102 + t * (182 - 102)), int(241 + t * (212 - 241)))
        d = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype("DejaVuSans-Bold.ttf", 150)
        except Exception:
            f = ImageFont.load_default()
        d.text((S / 2, S / 2), "AK", font=f, fill="white", anchor="mm")
        img.save(out)
    return str(out)
