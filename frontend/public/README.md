# frontend/public

Static assets copied verbatim into the build output at the site root.

## Favicons

`favicon.svg` is the source of truth, the seamed tennis-ball ball-mark. The two
raster fallbacks are **generated from it** and must be regenerated if that design
ever changes, or they will silently drift and look like a bug:

- `favicon.ico`, multi-resolution (16×16 + 32×32 + 48×48), transparent corners;
  fallback for browsers/contexts without SVG-favicon support.
- `apple-touch-icon.png`, 180×180, opaque square (iOS masks the corners itself);
  iOS home-screen / bookmark icon.

The order and attributes of the `<link>` tags in `index.html` matter (see the
["favicon nightmare"](https://dev.to/masakudamatsu/favicon-nightmare-how-to-maintain-sanity-3al7)
write-up): the `.ico` is listed **before** the `.svg`, `sizes="any"` goes on the
**SVG** (not the `.ico`), and the `.ico` carries `sizes="48x48"`. Chrome's rule is
"the last equally-appropriate icon wins," so that combination makes Chrome render
the crisp SVG while Safari/IE still fall back to the `.ico`. Inverting it makes
Chrome pick the raster fallback instead.

Regenerate (macOS, using QuickLook to rasterize + Pillow to resize, no npm
devDependency):

```bash
cd frontend/public
# 1. rasterize the SVG to a crisp 512px master (scale width/height to 512 first)
sed -E 's/width="48"/width="512"/;s/height="48"/height="512"/' favicon.svg > /tmp/favicon-512.svg
qlmanage -t -s 512 -o /tmp /tmp/favicon-512.svg          # -> /tmp/favicon-512.svg.png
# 2. white bg -> transparent (ico) / bg fill (apple-touch), then resize
python3 - <<'PY'
from PIL import Image
m = Image.open("/tmp/favicon-512.svg.png").convert("RGBA"); px = m.load(); W,H = m.size
trans = Image.new("RGBA",(W,H),(0,0,0,0)); tp = trans.load()
dark  = Image.new("RGBA",(W,H),(10,10,11,255)); dp = dark.load()
for y in range(H):
    for x in range(W):
        r,g,b,a = px[x,y]
        if r>240 and g>240 and b>240:       # QuickLook's white background
            tp[x,y]=(0,0,0,0); dp[x,y]=(10,10,11,255)
        else:
            tp[x,y]=(r,g,b,a); dp[x,y]=(r,g,b,255)
dark.convert("RGB").resize((180,180), Image.LANCZOS).save("apple-touch-icon.png")
# 48x48 must be present: index.html declares sizes="48x48" on the .ico.
trans.save("favicon.ico", format="ICO", sizes=[(16,16),(32,32),(48,48)])
PY
```

`favicon.svg` (opaque `#0a0a0b` rounded rect, `#d8ff3d` ball) needs no white-bg
step and is edited by hand as the design source.
