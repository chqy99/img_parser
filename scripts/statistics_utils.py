from typing import Tuple
from core.entity.image_parse_data import BBox


class StatisticsUtils:
    @staticmethod
    def compute_iou(box1: "BBox", box2: "BBox") -> float:
        """
        计算两个 BBox 的交并比 IoU
        """
        x_left = max(box1.x1, box2.x1)
        y_top = max(box1.y1, box2.y1)
        x_right = min(box1.x2, box2.x2)
        y_bottom = min(box1.y2, box2.y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = box1.area()
        box2_area = box2.area()
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    @staticmethod
    def is_bbox_covered_by_other(
        bbox1: "BBox", bbox2: "BBox", coverage_threshold: float = 0.9
    ) -> bool:
        """
        判断两个 bbox 中面积较小的那个 bbox 被另一个 bbox 覆盖的比例是否达到阈值。

        :param bbox1: 第一个 bbox
        :param bbox2: 第二个 bbox
        :param coverage_threshold: 覆盖比例阈值，默认0.9
        :return: True 如果较小 bbox 的被覆盖比例 >= coverage_threshold，否则 False
        """

        # 确定较小的 bbox 和较大的 bbox
        if bbox1.area() < bbox2.area():
            small_bbox, large_bbox = bbox1, bbox2
        else:
            small_bbox, large_bbox = bbox2, bbox1

        # 计算交集
        x_left = max(small_bbox.x1, large_bbox.x1)
        y_top = max(small_bbox.y1, large_bbox.y1)
        x_right = min(small_bbox.x2, large_bbox.x2)
        y_bottom = min(small_bbox.y2, large_bbox.y2)

        if x_right < x_left or y_bottom < y_top:
            # 无交集
            return False

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        small_area = small_bbox.area()
        if small_area == 0:
            # 防止除零错误
            return False

        coverage_ratio = intersection_area / small_area
        return coverage_ratio >= coverage_threshold
