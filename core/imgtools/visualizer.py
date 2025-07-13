### img_parser/core/imgtools/visualizer.py
import html
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Optional
from PIL import Image, ImageDraw
from core.imgdata.image_data import ImageParseUnit, ImageParseResult

# ======================== Palette ========================


def generate_even_palette(n):
    cm = plt.get_cmap("hsv")
    colors = [cm(i / n) for i in range(n)]
    return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]


def generate_random_palette(n, seed=42):
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 255, size=(n, 3)).tolist()


# ======================== Drawing Utils ========================


def apply_color_mapping(
    gray_image: np.ndarray, num_categories, palette=None, alpha=150
):
    if num_categories == 0 or gray_image is None:
        return None
    if palette is None:
        _palette = generate_even_palette(math.ceil(num_categories / 8) * 8)
    else:
        _palette = palette
    colored_image = np.zeros(
        (gray_image.shape[0], gray_image.shape[1], 4), dtype=np.uint8
    )
    for idx in range(1, num_categories + 1):
        if idx < len(_palette):
            color = _palette[idx - 1] + (alpha,)
            colored_image[gray_image == idx] = color
    return Image.fromarray(colored_image, mode="RGBA")


def apply_color_mapping_to_masks(masks: list[np.ndarray], palette=None, alpha=150):
    if not masks:
        return None
    height, width = masks[0].shape[:2]
    _palette = palette if palette is not None else generate_even_palette(len(masks))
    colored_image = np.zeros((height, width, 4), dtype=np.uint8)
    for i, mask in enumerate(masks):
        if i < len(_palette):
            color = _palette[i] + (alpha,)
            colored_image[mask] = color
    return colored_image


def apply_color_mapping_to_bboxs(
    image: np.ndarray,
    bboxs: list[list],
    color=(255, 0, 0),
    is_composite=True,
    alpha=150,
):
    pil_image = Image.fromarray(image)
    overlay = Image.new("RGBA", pil_image.size)
    draw = ImageDraw.Draw(overlay)
    for bbox in bboxs:
        draw.rectangle(
            [bbox[0], bbox[1], bbox[2], bbox[3]], outline=color + (alpha,), width=2
        )
    if is_composite:
        vis_image = pil_image.convert("RGBA")
        return Image.alpha_composite(vis_image, overlay)
    else:
        return overlay


def composite_overlap_image(
    base_img: np.ndarray, overlay_img: np.ndarray
) -> Image.Image:
    base_pil = Image.fromarray(base_img)
    overlay_pil = Image.fromarray(overlay_img)
    if base_pil.mode != "RGBA":
        base_pil = base_pil.convert("RGBA")
    if overlay_pil.mode != "RGBA":
        overlay_pil = overlay_pil.convert("RGBA")
    return Image.alpha_composite(base_pil, overlay_pil)


# ======================== Visualizer ========================


def visualize_parse_item(
    item: ImageParseUnit,
    show_mask: bool = True,
    show_bbox: bool = True,
    palette: Optional[List] = None,
    alpha: int = 150,
) -> Image.Image:
    base = item.image
    overlays = []
    if show_mask and item.mask is not None:
        mask_img = apply_color_mapping_to_masks(
            [item.mask], palette=palette, alpha=alpha
        )
        overlays.append(mask_img)
    if show_bbox and item.bbox is not None:
        bbox_img = apply_color_mapping_to_bboxs(
            base, [item.bbox.to_list()], is_composite=False, alpha=alpha
        )
        overlays.append(np.array(bbox_img))
    composite = Image.fromarray(base).convert("RGBA")
    for overlay in overlays:
        composite = composite_overlap_image(np.array(composite), overlay)
    return composite


def visualize_parse_result(
    result: ImageParseResult,
    show_mask: bool = True,
    show_bbox: bool = True,
    palette: Optional[List] = None,
    alpha: int = 150,
) -> Image.Image:
    base = result.image
    masks = (
        [item.mask for item in result.items if item.mask is not None]
        if show_mask
        else []
    )
    bboxs = (
        [item.bbox.to_list() for item in result.items if item.bbox is not None]
        if show_bbox
        else []
    )
    overlay = Image.new("RGBA", (base.shape[1], base.shape[0]))
    if show_mask and masks:
        mask_overlay = apply_color_mapping_to_masks(masks, palette=palette, alpha=alpha)
        overlay = composite_overlap_image(np.array(overlay), mask_overlay)
    if show_bbox and bboxs:
        bbox_overlay = apply_color_mapping_to_bboxs(
            base, bboxs, is_composite=False, alpha=alpha
        )
        overlay = composite_overlap_image(np.array(overlay), np.array(bbox_overlay))
    final = composite_overlap_image(base, np.array(overlay))
    return final


def generate_html_for_result(
    result: ImageParseResult, show_fields: List[str] = None
) -> str:
    if show_fields is None:
        show_fields = [
            "source_module",
            "score",
            "type",
            "label",
            "text",
            "bbox_image",
            "mask_image",
        ]

    headers_html = "".join(f"<th>{html.escape(field)}</th>" for field in show_fields)

    rows_html = ""
    for item in result.units:
        item_dict = item.to_dict(["bbox_image", "mask_image"])

        cells = []
        for field in show_fields:
            val = item_dict.get(field, None)
            if val is None:
                cells.append("<td></td>")
            elif field in ("bbox_image", "mask_image"):
                cells.append(
                    f'<td><img src="data:image/png;base64,{val}" style="max-width:120px;"/></td>'
                )
            elif isinstance(val, list):
                # 不截断，直接全部显示，支持换行
                full_val = ", ".join(map(str, val))
                cells.append(
                    f'<td style="word-break: break-word; white-space: pre-wrap;">{html.escape(full_val)}</td>'
                )
            else:
                # 普通文本支持换行
                cells.append(
                    f'<td style="word-break: break-word; white-space: pre-wrap;">{html.escape(str(val))}</td>'
                )

        rows_html += "<tr>" + "".join(cells) + "</tr>"

    html_str = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>ImageParseResult Visualization</title>
    <style>
    body {{ font-family: Arial, sans-serif; padding: 10px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; vertical-align: middle; }}
    th {{ background-color: #f2f2f2; }}
    img {{ max-width: 120px; height: auto; }}
    td {{ max-width: 400px; overflow-wrap: break-word; }}
    </style>
    </head>
    <body>
    <h1>ImageParseResult Visualization</h1>
    <table>
    <thead><tr>{headers_html}</tr></thead>
    <tbody>
    {rows_html}
    </tbody>
    </table>
    </body>
    </html>
    """
    return html_str
