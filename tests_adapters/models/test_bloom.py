from tests.models.bloom.test_modeling_bloom import *
from transformers import BloomAdapterModel
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class BloomAdapterModelTest(AdapterModelTesterMixin, BloomModelTest):
    all_model_classes = (BloomAdapterModel,)
    fx_compatible = False
