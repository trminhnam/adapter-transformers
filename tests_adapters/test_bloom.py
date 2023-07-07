import unittest
from tests_adapters.methods.test_config_union import ConfigUnionAdapterTest

from transformers import BloomConfig
from transformers.testing_utils import require_torch

from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
)
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .composition.test_parallel import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class BloomAdapterTestBase(AdapterTestBase):
    config_class = BloomConfig
    config = make_config(
        BloomConfig,
        n_embd=32,
        n_layer=4,
        n_head=4,
        # set pad token to eos token
        pad_token_id=3,
    )
    tokenizer_name = "bigscience/bloom-560m"


@require_torch
class BloomAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    EmbeddingTestMixin,
    CompabilityTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    ConfigUnionAdapterTest,
    BloomAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class BloomClassConversionTest(
    ModelClassConversionTestMixin,
    BloomAdapterTestBase,
    unittest.TestCase,
):
    pass
