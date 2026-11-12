"""Screenshots of Shinka's portal for one archived run: the program tree, the score trace and the best path.

Usage: python capture.py <db_path relative to the portal root> <out dir> <score axis minimum>
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

CHROME = Path.home() / ".cache/ms-playwright/chromium-1134/chrome-linux/chrome"
db, out, ymin = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
URL = f"http://127.0.0.1:8765/viz_tree.html?db_path={db}"


def page(p, width=1600, height=1000):
    b = p.chromium.launch(executable_path=str(CHROME), args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
    pg.goto(URL, timeout=120000)
    pg.wait_for_timeout(8000)
    return b, pg


with sync_playwright() as p:
    # the program tree: the left panel alone, full width; the best program's line is orange
    b, pg = page(p, 1800, 900)
    pg.evaluate("""() => {document.getElementById('details-panel').style.display = 'none';
                          const t = document.getElementById('tree-panel'); t.style.width = '100%'; t.style.flex = '1 1 100%';
                          window.dispatchEvent(new Event('resize'));}""")
    pg.wait_for_timeout(5000)
    pg.locator("#tree").screenshot(path=out / "tree.png")
    b.close()

    # the score trace: the Meta tab's first chart, its score axis from ymin
    b, pg = page(p)
    pg.evaluate("""(ymin) => {const h = [...document.querySelectorAll('#agent-info h4')].find(e => e.innerText.startsWith('Performance Score Across Program'));
                              const box = h.parentElement.querySelector('input'); box.value = ymin;
                              for (const t of ['input', 'change']) box.dispatchEvent(new Event(t, {bubbles: true}));
                              box.dispatchEvent(new KeyboardEvent('keyup', {key: 'Enter', bubbles: true}));}""", ymin)
    pg.wait_for_timeout(3000)
    box = pg.evaluate("""() => {const h = [...document.querySelectorAll('#agent-info h4')].find(e => e.innerText.startsWith('Performance Score Across Program'));
                                h.scrollIntoView(); const c = document.getElementById('scoreChart');
                                const a = h.getBoundingClientRect(), z = c.getBoundingClientRect();
                                return {x: Math.min(a.x, z.x) - 10, y: a.y - 8, width: Math.max(a.right, z.right) - Math.min(a.x, z.x) + 20, height: z.bottom - a.y + 4};}""")
    pg.screenshot(path=out / "score.png", clip=box)

    # the best path: the Path tab, grown to its full height on a white page
    pg.locator("div.left-tab", has_text="Path").first.click()
    pg.wait_for_timeout(4000)
    pg.evaluate("""() => {document.body.style.background = 'white';
                          for (const id of ['container', 'tree-panel', 'left-tab-content']) {const e = document.getElementById(id);
                          e.style.height = 'auto'; e.style.maxHeight = 'none'; e.style.overflow = 'visible'; e.style.background = 'white';}}""")
    pg.wait_for_timeout(1000)
    pg.locator("#left-tab-content").screenshot(path=out / "path.png")
    b.close()
print(sorted(f.name for f in out.iterdir()))
