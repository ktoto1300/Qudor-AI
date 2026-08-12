"""Measures where a horizontal wall is actually drawn, in a real browser.

The bar is a ::after pseudo-element, so its position cannot be asserted from the source - it
depends on the height the slot button ends up with, which is what the bug was about. This
loads index.html, renders a board, and compares the bar's centre against the centre of the
groove between the two rows of cells it is supposed to sit in.

Run: PYTHONIOENCODING=utf-8 python tools/wall_render_check.py
"""
import pathlib
import sys

from playwright.sync_api import sync_playwright

URL = pathlib.Path(__file__).resolve().parent.parent / "index.html"

# Row 3, column 3: a horizontal wall in the groove between cell rows 3 and 4.
R, C = 3, 3
H_SLOT = f"Горизонтальная стена {R + 1}, {C + 1}"

STATE = {
    "legal": [67, 75, 77], "hc": [], "hcOwner": [], "vc": [], "vcOwner": [],
    "legalWallsH": list(range(64)), "legalWallsV": list(range(64)),
    "p0": 76, "p1": 4, "mode": "human", "player": 0, "thinking": False,
}

# Re-creates the bug: the global `button{min-height:44px}` rule leaking into slots and cells.
BUG_CSS = ".wall-slot{min-height:44px}.cell{min-height:44px}"

MEASURE = """(label) => {
  const slot = document.querySelector(`[aria-label="${label}"]`);
  const cells = [...document.querySelectorAll('.cell')];
  const above = cells[3 * 9 + 3].getBoundingClientRect();
  const below = cells[4 * 9 + 3].getBoundingClientRect();
  const box = slot.getBoundingClientRect();
  const bar = getComputedStyle(slot, '::after');
  const grooveCentre = (above.bottom + below.top) / 2;
  // top is a used value in px here, and the bar is pulled up by half its own height.
  const barCentre = box.top + parseFloat(bar.top);
  const hit = (dy) => {
    const el = document.elementFromPoint(box.left + box.width / 2, grooveCentre + dy);
    return el ? (el.getAttribute('aria-label') || el.className) : 'none';
  };
  let reach = 0;
  while (reach < 40 && hit(reach + 1).startsWith('Горизонтальная')) reach++;
  return {
    slotHeight: box.height,
    grooveHeight: below.top - above.bottom,
    cellHeight: above.height,
    offset: barCentre - grooveCentre,
    onGroove: hit(0),
    intoCellBelow: hit((below.top - above.bottom) / 2 + 14),
    reachDown: reach,
  };
}"""


def measure(page, extra_css=None):
    page.goto(URL.as_uri())
    if extra_css:
        page.add_style_tag(content=extra_css)
    page.evaluate(
        "(s) => { state = s; inputMode = 'h'; renderBoard(); }", STATE
    )
    return page.evaluate(MEASURE, H_SLOT)


def placed_wall_blocks_cell(page):
    """A placed wall sits above the cells; it must not steal their clicks."""
    page.goto(URL.as_uri())
    page.evaluate(
        "(s) => { state = {...s, hc: [3 * 8 + 3], hcOwner: [0]}; inputMode = 'pawn';"
        " renderBoard(); }", STATE
    )
    return page.evaluate("""() => {
      const cells = [...document.querySelectorAll('.cell')];
      const below = cells[4 * 9 + 3].getBoundingClientRect();
      const el = document.elementFromPoint(below.left + below.width / 2, below.top + 6);
      return el ? (el.getAttribute('aria-label') || el.className) : 'none';
    }""")


def show(title, m):
    print(f"\n{title}")
    print(f"  высота слота          {m['slotHeight']:.0f}px "
          f"(щель {m['grooveHeight']:.0f}px, клетка {m['cellHeight']:.0f}px)")
    print(f"  полоска смещена на    {m['offset']:+.0f}px от центра щели")
    print(f"  клик в центр щели     -> {m['onGroove']}")
    print(f"  клик внутрь клетки    -> {m['intoCellBelow']}")
    print(f"  слот ловит клики до   {m['reachDown']}px вниз от щели")


with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.route("**/api/**", lambda route: route.abort())

    buggy = measure(page, BUG_CSS)
    fixed = measure(page)  # reloads without the injected bug, so vert below sees the fix too
    # Vertical slots span three tracks, so min-height never reached them - but they gained the
    # same click pad, and that must not have shifted the bar either.
    vert = page.evaluate("""(label) => {
      const slot = document.querySelector(`[aria-label="${label}"]`);
      const cells = [...document.querySelectorAll('.cell')];
      const left = cells[3 * 9 + 3].getBoundingClientRect();
      const right = cells[3 * 9 + 4].getBoundingClientRect();
      const box = slot.getBoundingClientRect();
      const bar = getComputedStyle(slot, '::after');
      return box.left + parseFloat(bar.left) - (left.right + right.left) / 2;
    }""", f"Вертикальная стена {R + 1}, {C + 1}")
    blocker = placed_wall_blocks_cell(page)
    browser.close()

show("ДО правки (min-height:44px возвращён):", buggy)
show("ПОСЛЕ правки:", fixed)
print(f"\nклик по клетке под поставленной стеной -> {blocker}")
print(f"вертикальная стена смещена на {vert:+.0f}px от центра щели")

ok = (abs(fixed["offset"]) < 1 and abs(vert) < 1
      and blocker.startswith("Клетка") and fixed["reachDown"] >= 8)
print("\n" + ("ВСЁ СХОДИТСЯ" if ok else "ЕСТЬ ПРОБЛЕМА"))
sys.exit(0 if ok else 1)
