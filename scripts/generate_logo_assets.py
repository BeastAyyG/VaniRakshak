import os
from PIL import Image, ImageDraw, ImageFilter

SOURCE_PATH = r"C:\Users\saumya\.gemini\antigravity-ide\brain\028a227d-d1b7-4b55-b19a-3a0f9e8ce928\.user_uploaded\media_1789161531499.jpg"
RES_DIR = r"C:\Users\saumya\VaniRakshak\vanirakshak-android\app\src\main\res"

def create_squircle_mask(size, radius):
    """Creates an anti-aliased rounded rectangle mask."""
    factor = 4  # supersample for ultra-smooth edges
    big_size = (size[0] * factor, size[1] * factor)
    mask = Image.new("L", big_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), (big_size[0] - 1, big_size[1] - 1)], radius=radius * factor, fill=255)
    return mask.resize(size, Image.Resampling.LANCZOS)

def create_circular_mask(size):
    """Creates an anti-aliased circular mask."""
    factor = 4
    big_size = (size[0] * factor, size[1] * factor)
    mask = Image.new("L", big_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([(0, 0), (big_size[0] - 1, big_size[1] - 1)], fill=255)
    return mask.resize(size, Image.Resampling.LANCZOS)

def main():
    print(f"Loading source logo from: {SOURCE_PATH}")
    src_img = Image.open(SOURCE_PATH).convert("RGBA")
    w, h = src_img.size

    # 1. Extract the squircle badge
    # The bounding box is roughly (51, 51, 971, 969) -> size 920x918
    crop_box = (51, 51, 971, 969)
    cropped_badge = src_img.crop(crop_box)
    badge_w, badge_h = cropped_badge.size
    
    # Apply rounded squircle mask to remove the outer black corners cleanly
    # Corner radius ~19% of badge dimension
    radius = int(badge_w * 0.19)
    mask = create_squircle_mask((badge_w, badge_h), radius)
    cropped_badge.putalpha(mask)

    # 2. Generate Primary In-App Logo: drawable/ic_logo_vanirakshak.png (512x512)
    drawable_dir = os.path.join(RES_DIR, "drawable")
    os.makedirs(drawable_dir, exist_ok=True)
    
    app_logo = cropped_badge.resize((512, 512), Image.Resampling.LANCZOS)
    app_logo_path = os.path.join(drawable_dir, "ic_logo_vanirakshak.png")
    app_logo.save(app_logo_path, "PNG")
    print(f"Saved: {app_logo_path}")

    # 3. Generate Splash Screen Logo: drawable/ic_logo_vanirakshak_splash.png (512x512)
    splash_logo = cropped_badge.resize((512, 512), Image.Resampling.LANCZOS)
    splash_logo_path = os.path.join(drawable_dir, "ic_logo_vanirakshak_splash.png")
    splash_logo.save(splash_logo_path, "PNG")
    print(f"Saved: {splash_logo_path}")

    # 4. Generate Adaptive Icon Foreground (drawable/ic_launcher_foreground.png)
    # Android adaptive icon standard: 432x432 canvas, safe zone is inner 66% (285px diameter).
    # We place the badge at 302x302 centered to ensure the shield, mic, and text never get clipped.
    adaptive_canvas_size = 432
    adaptive_fg = Image.new("RGBA", (adaptive_canvas_size, adaptive_canvas_size), (0, 0, 0, 0))
    fg_badge_size = 302
    scaled_fg_badge = cropped_badge.resize((fg_badge_size, fg_badge_size), Image.Resampling.LANCZOS)
    
    offset_x = (adaptive_canvas_size - fg_badge_size) // 2
    offset_y = (adaptive_canvas_size - fg_badge_size) // 2
    adaptive_fg.paste(scaled_fg_badge, (offset_x, offset_y), scaled_fg_badge)
    
    fg_path = os.path.join(drawable_dir, "ic_launcher_foreground.png")
    adaptive_fg.save(fg_path, "PNG")
    print(f"Saved Adaptive Foreground: {fg_path}")

    # 5. Generate Legacy Mipmap Icons for all Android densities
    densities = {
        "mipmap-mdpi": 48,
        "mipmap-hdpi": 72,
        "mipmap-xhdpi": 96,
        "mipmap-xxhdpi": 144,
        "mipmap-xxxhdpi": 192
    }

    for folder, size in densities.items():
        out_dir = os.path.join(RES_DIR, folder)
        os.makedirs(out_dir, exist_ok=True)

        # Standard squircle launcher icon
        icon_standard = cropped_badge.resize((size, size), Image.Resampling.LANCZOS)
        icon_path = os.path.join(out_dir, "ic_launcher.png")
        icon_standard.save(icon_path, "PNG")

        # Round launcher icon (circular mask)
        icon_round = cropped_badge.resize((size, size), Image.Resampling.LANCZOS)
        round_mask = create_circular_mask((size, size))
        icon_round.putalpha(round_mask)
        icon_round_path = os.path.join(out_dir, "ic_launcher_round.png")
        icon_round.save(icon_round_path, "PNG")

        print(f"Saved {folder}: {size}x{size}")

    print("All VaniRakshak branding assets successfully generated!")

if __name__ == "__main__":
    main()
