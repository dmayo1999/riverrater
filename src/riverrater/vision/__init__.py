"""
Vision package for RiverRater.

Exports the two detection engines and the calibration session class so that
consumers can do::

    from riverrater.vision import TemplateEngine, YOLOEngine, CalibrationSession
"""

from riverrater.vision.template_engine import TemplateEngine
from riverrater.vision.yolo_engine import YOLOEngine
from riverrater.vision.calibration import CalibrationSession

__all__ = [
    "TemplateEngine",
    "YOLOEngine",
    "CalibrationSession",
]
