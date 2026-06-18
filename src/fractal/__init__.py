"""Core fractal-dimension engine.

This package contains the scientific heart of the study:

* ``box_counting`` — the box-counting algorithm and log-log regression that
  estimate the fractal dimension D of a rasterized coastline segment.
* ``fractals``     — generators for fractal curves with *known* theoretical
  dimensions (Koch, Minkowski, straight line) used to validate the engine.
"""

from .box_counting import BoxCountResult, box_counting_dimension, count_boxes

__all__ = ["BoxCountResult", "box_counting_dimension", "count_boxes"]
