import cv2
import numpy as np


class ProcessUtils:
    @staticmethod
    def morphological_open(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        开运算，去噪声
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return opened

    @staticmethod
    def morphological_close(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        闭运算，填充小孔洞
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return closed

    @staticmethod
    def erode(
        image: np.ndarray, kernel_size: int = 3, iterations: int = 1
    ) -> np.ndarray:
        """
        腐蚀操作，缩小前景区域，去除边界噪声
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        eroded = cv2.erode(image, kernel, iterations=iterations)
        return eroded

    @staticmethod
    def dilate(
        image: np.ndarray, kernel_size: int = 3, iterations: int = 1
    ) -> np.ndarray:
        """
        膨胀操作，扩大前景区域，填充空洞
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilated = cv2.dilate(image, kernel, iterations=iterations)
        return dilated
