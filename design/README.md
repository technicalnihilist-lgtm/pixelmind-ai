# UI renders

These PNGs are **renders of the Compose layout**, produced from HTML that uses the same design
tokens as `ui/theme/` (colours, radii, type scale, glass recipe). They are not device
screenshots — no Android hardware was available when this repository was assembled.

| File | Screen |
| --- | --- |
| `01-home.png` | Home — prompt, style presets, recent creations, local AI status |
| `02-studio.png` | Studio — negative prompt, aspect, quality tiers, live progress |
| `03-gallery.png` | Gallery — grid, favourites, prompt history |
| `04-settings.png` | Settings — model, hardware, storage, appearance, privacy |
| `05-sample-output.png` | Actual output from the shipped verification model (512×512, 4 steps, seed 1234) |

The `.html` files next to each PNG are the sources; open one in a browser to see it live. The
abstract gradient tiles stand in for generated images — they are illustrative, not model output.
