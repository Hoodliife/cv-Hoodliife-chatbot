# Christmas wreath – embroidery files

| File | Size | Hoop |
|---|---|---|
| `christmas-wreath-100mm.pes` | 94 × 100 mm | 4×4" (100×100) |
| `christmas-wreath-130mm.pes` | 123 × 130 mm | 5×7" (130×180) |

**2 threads, 1 colour change**

1. **Green** (#2E7D32): every green element (dark, mid and light leaves, pine sprigs, dots) in a single thread
2. **Red** (#C62828): bow and berries

Digitizing spec (premium):
- **Leaves**: true satin, 0.32 mm density, angled 28°, one column per leaf half (midrib split like the original), edge-walk + zigzag underlay, 0.15 mm pull compensation
- **Berries**: each berry its own round satin with underlay
- **Bow**: satin column that follows the ribbon, centre-run underlay, width tapers with the ribbon
- **Pine sprigs**: bean-stitched (triple) stem with a fine needle every 0.38 mm
- No stitch longer than 7 mm; tie-in/tie-off at every trim

Stitch order: pine sprigs → leaves → berries → bow. Stitch out a test piece on scrap fabric with stabilizer before the final piece.
The tiny dark berry "eyes" in the photo are filled red. Stitch them by hand or add a 3rd colour if you want them.

Regenerate with: `python3 digitize.py source-image.jpg . 100 130` (needs `pyembroidery opencv-python-headless scikit-image numpy`). `render_preview.py` makes the thread-realistic previews.
