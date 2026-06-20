# Real data input (image / "real" mode)

To run the pipeline on your own coastline maps instead of synthetic data:

1. **Set the mode.** In `config.yaml`, set `mode: real`.

2. **Add your coastline images.** Put **one JPG per segment** in
   `data/raw/images/`. Each image should be a *filled land-vs-sea* map: land one
   shade/colour, sea another. The pipeline extracts the **land/sea boundary**
   (the coastline) automatically — it does not matter which side is land.
   - Supported: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`
   - Images are processed in **filename order**, so name them so they sort the
     way you want, e.g. `seg_01.jpg`, `seg_02.jpg`, …

3. **Describe each segment.** Fill in `data/raw/segments.csv` (template
   provided). It is matched to images by the `filename` column:

   | column           | required | meaning                                            |
   | ---------------- | -------- | -------------------------------------------------- |
   | `filename`       | yes      | image file name in `data/raw/images/`              |
   | `surge_height_m` | see below| explicit max surge height (m) for this segment      |
   | `ssa_level`      | see below| PAGASA SSA level: `none`, `1`, `2`, `3`, or `4`    |
   | `location`       | no       | place name (defaults to the filename)              |
   | `province`       | no       | province                                           |
   | `region`         | no       | region                                             |
   | `lat`, `lon`     | no       | centroid coordinates (used for the map)            |
   | `id`             | no       | segment id (defaults to 01, 02, …)                 |

   **Surge value — give each segment ONE of these:**
   - `surge_height_m`: an exact height in metres (use this if you have a recorded
     or modelled value), **or**
   - `ssa_level`: the PAGASA Storm Surge Advisory level from
     [HazardHunterPH](https://hazardhunter.georisk.gov.ph/map). It is mapped to a
     representative height: `none`→0.5 m, `1`→2.5 m, `2`→3.5 m, `3`→4.5 m,
     `4`→5.5 m. (SSA bands: 1 = 2–3 m, 2 = 3–4 m, 3 = 4–5 m, 4 = >5 m.)

   If a segment has `surge_height_m`, that wins; otherwise its `ssa_level` is used.
   Every segment needs one or the other, or the statistics stage will stop and
   tell you which are missing. Since SSA values are ordinal, **Spearman's rho**
   (already reported) is the most appropriate correlation test for them.

   > How to look up SSA in HazardHunterPH: open the map, search each segment's
   > location (or enter its lat/lon from this CSV), read the Storm Surge hazard
   > result, and record the level. There is no public API, so this step is manual.

4. **Run it.** `python main.py` — outputs land in `output/tables`,
   `output/reports`, and `output/figures` exactly as in synthetic mode.

## Tuning extraction (config.yaml -> `image:`)

If the coastline comes out noisy or incomplete, adjust:

- `threshold`: `otsu` (automatic) or a fixed gray level `0–255`.
- `keep_largest`: keep only the largest region (drops text/legend specks).
- `fill_holes`: fill interior holes (lakes, labels printed over land).
- `crop_border_frac`: crop a fraction off each edge to remove map frames/margins.
