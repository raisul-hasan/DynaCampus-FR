# pyrefly: ignore [missing-import]
import pytest
import numpy as np
from src.augmentation.cctv_augmentor import CCTVAugmentor

def test_cctv_augmentor_generate_variants():
    augmentor = CCTVAugmentor()
    img = np.uint8(np.random.rand(112, 112, 3) * 255)
    variants = augmentor.generate_cctv_variants(img)
    assert len(variants) == 5
    for v in variants:
        assert v.shape == (112, 112, 3)
