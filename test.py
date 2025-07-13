import sys
import os
import time
import numpy as np
from PIL import Image
from core.imgdata.image_data import ImageParseResult, ImageParseUnit
from core.modules.module_factory import ModuleFactory
from core.imgtools import visualizer

# ======================
# 自动注册测试函数
# ======================
tests = {}


def register_test(name):
    def decorator(func):
        tests[name] = func
        return func

    return decorator


# ======================
# 工具函数
# ======================


def ensure_log_dir():
    log_dir = "LOG"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def timestamp_str():
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def save_vis_image(result: ImageParseResult, path):
    vis_img = visualizer.visualize_parse_result(result, show_mask=True, show_bbox=True)
    vis_img.save(path)
    print(f"可视化图片已保存到: {path}")


def save_html(result: ImageParseResult, path):
    html_content = visualizer.generate_html_for_result(result)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML 文件已保存到: {path}")

def test_memory_func(result):
    from core.memory.image_memory import ImageMemory
    memory = ImageMemory(collection_name="test")
    # 依次存储
    if result.image is not None:
        memory.save_raw_image(result)
        print("已存储原图 embedding")
    if hasattr(result, "bboxs_image") and result.bboxs_image is not None:
        memory.save_result(result, type="bbox")
        print("已存储 bbox embedding")
    if hasattr(result, "masks_image") and result.masks_image is not None:
        memory.save_result(result, type="mask")
        print("已存储 mask embedding")
    # 检索
    retrieved = memory.query_result(getattr(result, "image", None), top_k=1)
    print("检索结果：")
    for i, item in enumerate(retrieved):
        print(f"[{i}]", item.get("uid", item))


def handle_output(
    result: ImageParseResult, output_mode: str = "print", output_path: str = None
):
    if output_mode == "print":
        print(result)
    elif output_mode == "img":
        path = output_path or os.path.join(ensure_log_dir(), f"{timestamp_str()}.png")
        save_vis_image(result, path)
    elif output_mode == "html":
        path = output_path or os.path.join(ensure_log_dir(), f"{timestamp_str()}.html")
        save_html(result, path)
    elif output_mode == "memory":
        test_memory_func(result)
    else:
        print(f"[警告] 不支持的输出模式: {output_mode}，默认打印结果")
        print(result)


# ======================
# 各测试函数
# ======================


@register_test("paddleocr")
def test_paddleocr(img_path, output_mode="print", output_path=None):
    import core.modules.paddleocr_module

    paddleocr_module = ModuleFactory.get_module("paddleocr")
    image = np.array(Image.open(img_path).convert("RGB"))
    result: ImageParseResult = paddleocr_module.parse(image)
    handle_output(result, output_mode, output_path)


@register_test("clip")
def test_clip(img_path, output_mode="print", output_path=None):
    import core.modules.clip_module

    clipModule = ModuleFactory.get_module("clip")
    image = np.array(Image.open(img_path))
    result = clipModule.parse([ImageParseUnit(image=image, bbox=None, source_module="")], filter="image")
    print(result)


@register_test("florence2")
def test_florence2(img_path, output_mode="print", output_path=None):
    import core.modules.florence2_module

    florence2Module = ModuleFactory.get_module("florence2")
    image = np.array(Image.open(img_path))
    result = florence2Module.parse([ImageParseUnit(image=image, bbox=None, source_module="")], filter="image")
    print(result)


@register_test("florence2_icon")
def test_florence2_icon(img_path, output_mode="print", output_path=None):
    import core.modules.florence2_module

    florence2Module = ModuleFactory.get_module("florence2_icon")
    image = np.array(Image.open(img_path))
    result = florence2Module.parse([ImageParseUnit(image=image, bbox=None, source_module="")], filter="image")
    print(result)


@register_test("sam2")
def test_sam2(img_path, output_mode="print", output_path=None):
    import core.modules.sam2_module

    sam_module = ModuleFactory.get_module("sam2")
    image = np.array(Image.open(img_path))
    result: ImageParseResult = sam_module.parse(image)
    handle_output(result, output_mode, output_path)


@register_test("yolo")
def test_yolo(img_path, output_mode="print", output_path=None):
    import core.modules.yolo_module

    yoloModule = ModuleFactory.get_module("yolo")
    image = np.array(Image.open(img_path).convert("RGB"))
    result = yoloModule.parse(image)
    handle_output(result, output_mode, output_path)


@register_test("custom_omni_parser")
def test_custom_omni_parser(img_path, output_mode="print", output_path=None):
    from core.pipeline.custom_omni_parser import CustomOmniParser

    omni_parser = CustomOmniParser()
    image = np.array(Image.open(img_path).convert("RGB"))
    result = omni_parser.parse(image)
    handle_output(result, output_mode, output_path)


@register_test("semantic_parser")
def test_semantic_parser(img_path, output_mode="print", output_path=None):
    from core.pipeline.semantic_parser import SemanticParser

    semantic_parser = SemanticParser()
    image = np.array(Image.open(img_path).convert("RGB"))
    result = semantic_parser.parse(image)
    handle_output(result, output_mode, output_path)


# ======================
# 主程序入口
# ======================


def main():
    if len(sys.argv) < 3:
        print("用法： python test.py <测试名> <图片路径> [输出模式] [输出路径]")
        print("输出模式可选： print（默认）、img、html")
        print("示例： python test.py paddleocr E:/img.png img")
        print("支持的测试名：")
        for name in tests:
            print(" -", name)
        return

    test_name = sys.argv[1]
    img_path = sys.argv[2]
    output_mode = sys.argv[3] if len(sys.argv) > 3 else "print"
    output_path = sys.argv[4] if len(sys.argv) > 4 else None

    if test_name not in tests:
        print(f"未知测试名：{test_name}")
        print("支持的测试名：")
        for name in tests:
            print(" -", name)
        return

    if not os.path.isfile(img_path):
        print(f"图片路径不存在或不是文件：{img_path}")
        return

    tests[test_name](img_path, output_mode, output_path)


if __name__ == "__main__":
    main()
