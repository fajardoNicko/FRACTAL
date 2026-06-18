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
   | `surge_height_m` | yes      | max PAGASA surge height (m) for this segment        |
   | `location`       | no       | place name (defaults to the filename)              |
   | `province`       | no       | province                                           |
   | `region`         | no       | region                                             |
   | `lat`, `lon`     | no       | centroid coordinates (used for the map)            |
   | `id`             | no       | segment id (defaults to 01, 02, …)                 |

   If `segments.csv` is missing, images are still processed, but `surge_height_m`
   is required for the correlation step — so the statistics stage will stop and
   tell you to add it.

4. **Run it.** `python main.py` — outputs land in `output/tables`,
   `output/reports`, and `output/figures` exactly as in synthetic mode.

## Tuning extraction (config.yaml -> `image:`)

If the coastline comes out noisy or incomplete, adjust:

- `threshold`: `otsu` (automatic) or a fixed gray level `0–255`.
- `keep_largest`: keep only the largest region (drops text/legend specks).
- `fill_holes`: fill interior holes (lakes, labels printed over land).
- `crop_border_frac`: crop a fraction off each edge to remove map frames/margins.
