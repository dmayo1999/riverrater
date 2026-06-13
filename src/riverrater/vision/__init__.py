"""
Vision package for RiverRater.

Exports the two detection engines and the calibration session class so that
consumers can do::

    from riverrater.vision import TemplateEngine, YOLOEngine, CalibrationSession
"""

from riverrater.vision.card_tracker import CardTracker
from riverrater.vision.template_engine import TemplateEngine
from riverrater.vision.yolo_engine import YOLOEngine
from riverrater.vision.calibration import CalibrationSession
from riverrater.vision.pot_ocr import PotOCR, PotOCRResult

__all__ = [
    "CardTracker",
    "TemplateEngine",
    "YOLOEngine",
    "CalibrationSession",
    "PotOCR",
    "PotOCRResult",
]
