#!/usr/bin/env python3
"""
Build PEBBL_User_Manual.pdf from PEBBL_User_Manual.html (the web-page manual).

The HTML is written as page content only (the web version adds the document
skeleton when it's published), so this wraps it in a full document, then
prints it with Microsoft Edge or Google Chrome in headless mode. The typefaces
come from Google Fonts and the logo is drawn by a script, so the browser is
given time to load and draw before printing (needs an internet connection).

Run from anywhere:  annotate_env\\Scripts\\python.exe docs\\build_manual_pdf.py
Optional:  --png  also saves a full-page PNG preview next to the PDF.
Also renders the logo icon to img/pebbl_icon.png (transparent background) for
the public repository's front page, from branding/pebbl_core.js.
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "PEBBL_User_Manual.html")
PDF = os.path.join(HERE, "PEBBL_User_Manual.pdf")
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]
# Print layout: US Letter with comfortable margins; the page's own print styles do the rest.
PRINT_CSS = "<style>@page{size:Letter;margin:0.6in 0.65in}</style>"


def browser():
    for path in BROWSERS:
        if os.path.exists(path):
            return path
    found = shutil.which("msedge") or shutil.which("google-chrome") or shutil.which("chromium")
    if not found:
        sys.exit("No Edge or Chrome found to print the PDF with.")
    return found


ICON_PNG = os.path.join(HERE, "img", "pebbl_icon.png")
# The Windows icon for desktop shortcuts to "Start PEBBL.bat" (HLU, 2026-10-05), in PEBBL's main folder.
ICON_ICO = os.path.join(os.path.dirname(HERE), "pebbl.ico")
ICO_SIZES = [(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]
CORE_JS = os.path.join(os.path.dirname(HERE), "branding", "pebbl_core.js")
ICON_PAGE = ("<!doctype html><html><head><meta charset='utf-8'>"
             "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@700&display=swap'>"
             "<style>html,body{margin:0;background:transparent}#i{width:480px;margin:40px}</style></head><body><div id='i'></div>"
             "<script>%s</script><script>var I={W:1732,H:2000,border:95,bleed:0,seed:61,tries:3500,minR:80,maxR:190,"
             "line:12,tw:100,halo:190,gap:40,fs:500,textTop:250,dipBelowTop:40,tipDrop:210};"
             "document.fonts.load(\"700 500px 'Source Sans 3'\").then(function(){PEBBL(document.getElementById('i'),I);});"
             "</script></body></html>")


def render_icon(common):
    with open(CORE_JS, encoding="utf-8") as f:
        page = ICON_PAGE % f.read()
    fd, temp_html = tempfile.mkstemp(suffix=".html", dir=HERE)
    os.close(fd)
    try:
        with open(temp_html, "w", encoding="utf-8") as f:
            f.write(page)
        subprocess.run(common + ["--window-size=640,720", "--hide-scrollbars", "--default-background-color=00000000",
                                 f"--screenshot={ICON_PNG}", "file:///" + temp_html.replace("\\", "/")],
                       check=True, capture_output=True, timeout=180)
        # Trim to the drawn hexagon (its alpha bounding box), then scale to 240 px wide.
        from PIL import Image
        image = Image.open(ICON_PNG).convert("RGBA")
        image = image.crop(image.getchannel("A").getbbox())
        # The .ico from the full-size render: centered on a transparent square (icons are square).
        side = max(image.size)
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        square.save(ICON_ICO, format="ICO", sizes=ICO_SIZES)
        image.resize((240, round(240 * image.height / image.width)), Image.LANCZOS).save(ICON_PNG)
        print("wrote", ICON_PNG)
        print("wrote", ICON_ICO)
    finally:
        os.remove(temp_html)


def main():
    with open(SOURCE, encoding="utf-8") as f:
        body = f.read()
    wrapped = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
               "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">" + PRINT_CSS +
               "</head><body>" + body + "</body></html>")
    # The wrapped copy sits next to the source so the relative img/ paths still resolve.
    fd, temp_html = tempfile.mkstemp(suffix=".html", dir=HERE)
    os.close(fd)
    profile = tempfile.mkdtemp(prefix="pebbl_manual_browser_")
    try:
        with open(temp_html, "w", encoding="utf-8") as f:
            f.write(wrapped)
        url = "file:///" + temp_html.replace("\\", "/")
        common = [browser(), "--headless", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}",
                  "--virtual-time-budget=15000", "--run-all-compositor-stages-before-draw"]
        subprocess.run(common + ["--no-pdf-header-footer", f"--print-to-pdf={PDF}", url], check=True,
                       capture_output=True, timeout=180)
        print("wrote", PDF)
        render_icon(common)
        if "--png" in sys.argv:
            png = os.path.join(HERE, "PEBBL_User_Manual_preview.png")
            subprocess.run(common + ["--window-size=900,9000", "--hide-scrollbars", f"--screenshot={png}", url],
                           check=True, capture_output=True, timeout=180)
            print("wrote", png)
    finally:
        os.remove(temp_html)
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
