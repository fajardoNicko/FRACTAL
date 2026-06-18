"""Preprocessing: turn vector coastline data into the binary images that the
box-counting engine consumes.

* ``raster_converter`` — rasterize a polyline (or shapely geometry) to a binary
  grid. A NumPy-only path is always available for validation; the geospatial
  path (geopandas/rasterio) is used once real NAMRIA data is supplied.
"""
