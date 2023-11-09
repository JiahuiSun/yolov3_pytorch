from .model import Darknet10, Darknet53
from .model_atten import Darknet53Atten


# 模型汇总
MODEL_REGISTRY = {
    'Darknet53': Darknet53,
    'Darknet10': Darknet10,
    'Darknet53Atten': Darknet53Atten
}
