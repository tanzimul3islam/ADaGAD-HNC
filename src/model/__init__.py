from src.model.adagad_hnc import AdaGADHNC, build_adagad_hnc
from src.model.discriminators import Discriminator
from src.model.encoders import ENCODER_REGISTRY, build_encoder
from src.model.fusion import AdaptiveFusion
from src.model.readout import READOUT_REGISTRY, build_readout
from src.model.reconstruction import GraphDecoder, MLPDecoder
from src.utils.registry import MODEL_REGISTRY

MODEL_REGISTRY.register("adagad_hnc")(AdaGADHNC)

__all__ = [
    "build_encoder",
    "build_readout",
    "Discriminator",
    "MLPDecoder",
    "GraphDecoder",
    "AdaptiveFusion",
    "AdaGADHNC",
    "build_adagad_hnc",
    "ENCODER_REGISTRY",
    "READOUT_REGISTRY",
    "MODEL_REGISTRY",
]
