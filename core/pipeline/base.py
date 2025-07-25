# core/pipeline/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np
from core.entity.image_parse_data import ImageParseResult
from core.modules.module_factory import ModuleFactory


class PipelineParser(ABC):
    def __init__(self, module_names: List[str]):
        """
        :param module_names: 模块名列表（已注册到 ModuleFactory 中）
        """
        self.module_names = module_names
        self.modules: Dict[str, Any] = {}

    def get_module(self, name: str):
        """获取模块实例，必要时从 ModuleFactory 构造"""
        if name not in self.modules:
            self.modules[name] = ModuleFactory.get_module(name)
        return self.modules[name]

    @abstractmethod
    def parse(self, image: np.ndarray, **kwargs) -> ImageParseResult:
        pass
