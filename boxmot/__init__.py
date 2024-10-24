# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '10.0.75'

from boxmot.postprocessing.gsi import gsi
from boxmot.tracker_zoo import create_tracker, get_tracker_config
from boxmot.trackers.botsort.bot_sort import BoTSORT
from boxmot.trackers.ftosort.fto_sort import FTOSORT
from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
from boxmot.trackers.deepocsort.deep_ocsort import DeepOCSort as DeepOCSORT
from boxmot.trackers.hybridsort.hybridsort import HybridSORT
from boxmot.trackers.ocsort.ocsort import OCSort as OCSORT
from boxmot.trackers.strongsort.strong_sort import StrongSORT
from boxmot.trackers.imprassoc.impr_assoc_tracker import ImprAssocTrack


TRACKERS = ['bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'hybridsort', 'imprassoc', 'fto_sort']

__all__ = ("__version__",
           "StrongSORT", "OCSORT", "BYTETracker", "BoTSORT", "DeepOCSORT", "HybridSORT", "ImprAssocTrack", "FTOSORT"
           "create_tracker", "get_tracker_config", "gsi")
