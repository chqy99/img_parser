# core/modules/__init__.py
import importlib
import pkgutil
from pathlib import Path

# 把当前包下所有 .py 文件都 import 一遍，触发装饰器注册
def _auto_import_all():
    package_dir = Path(__file__).parent
    for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        importlib.import_module(f"{__package__}.{module_name}")

_auto_import_all()
