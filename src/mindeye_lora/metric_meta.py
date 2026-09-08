"""Metric names, directions, and descriptions.

Kept free of torch so the analysis and reporting layers can be imported (and tested)
without a GPU stack.
"""
from __future__ import annotations

EMBEDDING_METRICS = ["cosine", "two_way_clip", "retrieval_percentile"]
IMAGE_METRICS = ["pixcorr", "ssim", "alexnet2", "alexnet5", "inception", "clip", "effnet", "swav"]
ALL_METRICS = EMBEDDING_METRICS + IMAGE_METRICS

HIGHER_IS_BETTER = {
    "cosine": True,
    "two_way_clip": True,
    "retrieval_percentile": True,
    "pixcorr": True,
    "ssim": True,
    "alexnet2": True,
    "alexnet5": True,
    "inception": True,
    "clip": True,
    "effnet": False,   # correlation distance
    "swav": False,     # correlation distance
}

DESCRIPTIONS = {
    "cosine": "cosine similarity between the predicted and true CLIP token embeddings",
    "two_way_clip": "per-image probability that the true image outranks a random "
                    "distractor in CLIP space",
    "retrieval_percentile": "rank of the correct image among all test images, rescaled "
                            "so 1.0 is a perfect top-1 retrieval",
    "pixcorr": "pixel-wise correlation between reconstruction and stimulus",
    "ssim": "structural similarity on greyscale images",
    "alexnet2": "two-way identification using AlexNet layer-2 features (low level)",
    "alexnet5": "two-way identification using AlexNet layer-5 features (mid level)",
    "inception": "two-way identification using InceptionV3 pooled features",
    "clip": "two-way identification in CLIP image-embedding space",
    "effnet": "EfficientNet-B1 correlation distance (lower is better)",
    "swav": "SwAV-ResNet50 correlation distance (lower is better)",
}

PRIMARY_METRIC = "two_way_clip"
