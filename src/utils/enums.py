from enum import Enum
from typing import Any
import cv2
from config import config


class DetectorType(Enum):
    """Enum for detector types used in feature extraction."""

    ORB = 0
    FAST = 1
    SIFT = 2


def create_detector(detector_type: DetectorType, **kwargs: float) -> cv2.Feature2D:
    """
    Create a feature detector based on the specified type.

    Args:
        detector_type: Type of detector to create.
        **kwargs: Additional arguments passed to the detector constructor.

    Returns:
        A cv2.Feature2D detector instance.

    Raises:
        ValueError: If the detector type is not supported.

    """
    if detector_type == DetectorType.FAST:
        return cv2.FastFeatureDetector.create(**kwargs)
    if detector_type == DetectorType.ORB:
        return cv2.ORB.create(**kwargs)
    if detector_type == DetectorType.SIFT:
        return cv2.SIFT.create(
            nfeatures=config.SIFT_NFEATURES,
            nOctaveLayers=config.SIFT_NOCTAVES,
            contrastThreshold=config.SIFT_CONTRASTTHRESH,
            edgeThreshold=config.SIFT_EDGETHRESH,
            sigma=config.SIFT_SIGMA,
            enable_precise_upscale=True,
        )
    msg = "Unsupported detector type"
    raise ValueError(msg)


class DescriptorType(Enum):
    """Enum for descriptor types used in feature extraction."""

    ORB = 0
    SIFT = 1
    FREAK = 2


def create_descriptor(
    descriptor_type: DescriptorType, **kwargs: float
) -> cv2.Feature2D:
    """
    Create a feature descriptor based on the specified type.

    Args:
        descriptor_type: Type of descriptor to create.
        **kwargs: Additional arguments passed to the descriptor constructor.

    Returns:
        A cv2.Feature2D descriptor instance.

    Raises:
        ValueError: If the descriptor type is not supported.

    """
    if descriptor_type == DescriptorType.ORB:
        return cv2.ORB.create(**kwargs)
    if descriptor_type == DescriptorType.SIFT:
        return cv2.SIFT.create(**kwargs)
    msg = "Unsupported descriptor type"
    raise ValueError(msg)
