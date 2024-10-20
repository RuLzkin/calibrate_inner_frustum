import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .module_load_exr import load_exr


@dataclass
class Point:
    x: int
    y: int


@dataclass
class ZoomInfo:
    center: Point
    top_left: Point
    width: int
    height: int


class ColorcheckerExtractor:
    ZOOM_FACTOR = 4
    MAX_DISPLAY_WIDTH = 1280
    MAX_DISPLAY_HEIGHT = 720
    IS_DEBUG = False

    def __init__(self, image: np.ndarray):
        self.original_image = image.copy()
        self.display_image = image.copy()
        self.coordinates: List[Point] = []
        self.is_zoomed = False
        self.zoom_info: Optional[ZoomInfo] = None
        self.scale_factor = 1.0
        self.zoom_scale_factor = 1.0

    def coords_dst(self):
        shape_trans = self.shape_transform()
        pts_dst = np.float32(
            [[0, 0], [shape_trans[0], 0], [0, shape_trans[1]], [shape_trans[0], shape_trans[1]]])
        return pts_dst

    def shape_transform(self):
        return int(self.original_image.shape[1]), int(self.original_image.shape[0])

    def print(self, *args, **kwargs):
        if self.IS_DEBUG:
            print(*args, **kwargs)

    def resize_image(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        if w > self.MAX_DISPLAY_WIDTH or h > self.MAX_DISPLAY_HEIGHT:
            scale_w = self.MAX_DISPLAY_WIDTH / w
            scale_h = self.MAX_DISPLAY_HEIGHT / h
            scale_factor = min(scale_w, scale_h)
            new_size = (int(w * scale_factor), int(h * scale_factor))
            return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), scale_factor
        return img.copy(), 1.0

    def add_instructions(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        instructions = [
            "Left click: Zoom / Select point",
            "Right click: Cancel zoom",
            "R: Reset selection",
            "ESC: Exit"
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        padding = 10
        line_height = 20

        for i, instruction in enumerate(instructions):
            text_size = cv2.getTextSize(instruction, font, font_scale, font_thickness)[0]
            y = h - padding - (len(instructions) - i - 1) * line_height
            x = w - text_size[0] - padding

            cv2.rectangle(img, (x - 5, y - text_size[1] - 5), (w - padding + 5, y + 5), (0, 0, 0), -1)
            cv2.putText(img, instruction, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        return img

    def handle_click(self, event: int, x: int, y: int) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_zoomed:
                self.add_point(x, y)
            else:
                self.zoom_and_display(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN and self.is_zoomed:
            self.cancel_zoom()

    def add_point(self, x: int, y: int) -> None:
        if self.is_zoomed and self.zoom_info:
            # Convert zoomed display coordinates to original image coordinates
            zoom_x_ratio = (x - self.display_image.shape[1] / 2) / (self.display_image.shape[1] / 2)
            zoom_y_ratio = (y - self.display_image.shape[0] / 2) / (self.display_image.shape[0] / 2)

            original_x = int(self.zoom_info.center.x + zoom_x_ratio * self.zoom_info.width / 2)
            original_y = int(self.zoom_info.center.y + zoom_y_ratio * self.zoom_info.height / 2)
        else:
            # Convert non-zoomed display coordinates to original image coordinates
            original_x = int(x / self.scale_factor)
            original_y = int(y / self.scale_factor)

        self.coordinates.append(Point(original_x, original_y))
        cv2.circle(self.original_image, (original_x, original_y), 5, (0, 255, 0), -1)

        self.print(f"Added point: display({x}, {y}) -> original({original_x}, {original_y})")
        if self.is_zoomed and self.zoom_info:
            self.print(
                f"Zoom info: center({self.zoom_info.center.x}, {self.zoom_info.center.y}), "
                f"top_left({self.zoom_info.top_left.x}, {self.zoom_info.top_left.y}), "
                f"width={self.zoom_info.width}, height={self.zoom_info.height}")
        self.print(f"Current scale factors: scale_factor={self.scale_factor}, zoom_scale_factor={self.zoom_scale_factor}")

        self.cancel_zoom()

    def zoom_and_display(self, center_x: int, center_y: int) -> None:
        zoom_center = Point(int(center_x / self.scale_factor), int(center_y / self.scale_factor))
        zoom_width = self.original_image.shape[1] // self.ZOOM_FACTOR
        zoom_height = self.original_image.shape[0] // self.ZOOM_FACTOR

        x_start = max(0, zoom_center.x - zoom_width // 2)
        y_start = max(0, zoom_center.y - zoom_height // 2)
        x_end = min(self.original_image.shape[1], x_start + zoom_width)
        y_end = min(self.original_image.shape[0], y_start + zoom_height)

        zoomed_region = self.original_image[y_start:y_end, x_start:x_end]

        # Resize the zoomed region to fit the display
        self.display_image, self.zoom_scale_factor = self.resize_image(cv2.resize(zoomed_region, None, fx=self.ZOOM_FACTOR, fy=self.ZOOM_FACTOR))
        self.scale_factor = self.zoom_scale_factor * self.ZOOM_FACTOR

        self.zoom_info = ZoomInfo(
            center=zoom_center,
            top_left=Point(x_start, y_start),
            width=x_end - x_start,
            height=y_end - y_start
        )

        self.display_image = self.add_instructions(self.display_image)
        cv2.imshow('Image', self.display_image)
        self.is_zoomed = True

        self.print(
            f"Zoomed to: center({zoom_center.x}, {zoom_center.y}), "
            f"top_left({x_start}, {y_start}), width={x_end-x_start}, height={y_end-y_start}")
        self.print(f"New scale factors: scale_factor={self.scale_factor}, zoom_scale_factor={self.zoom_scale_factor}")

    def cancel_zoom(self) -> None:
        self.is_zoomed = False
        self.zoom_info = None
        self.display_image, self.scale_factor = self.resize_image(self.original_image)
        self.zoom_scale_factor = 1.0
        self.display_image = self.add_instructions(self.display_image)
        cv2.imshow('Image', self.display_image)
        self.print("Zoom cancelled")
        self.print(f"Reset scale factors: scale_factor={self.scale_factor}, zoom_scale_factor={self.zoom_scale_factor}")

    def reset_selection(self) -> None:
        self.coordinates = []
        self.is_zoomed = False
        self.zoom_info = None
        self.display_image, self.scale_factor = self.resize_image(self.original_image)
        self.zoom_scale_factor = 1.0
        self.display_image = self.add_instructions(self.display_image)
        cv2.imshow('Image', self.display_image)
        self.print("Selection reset")
        self.print(f"Reset scale factors: scale_factor={self.scale_factor}, zoom_scale_factor={self.zoom_scale_factor}")

    def get_four_points(self) -> List[Point]:
        self.display_image, self.scale_factor = self.resize_image(self.original_image)
        self.display_image = self.add_instructions(self.display_image)
        cv2.imshow('Image', self.display_image)
        cv2.setMouseCallback('Image', lambda event, x, y, flags, param: self.handle_click(event, x, y))

        while len(self.coordinates) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('r'):
                self.reset_selection()

            if len(self.coordinates) > 0 and len(self.coordinates) % 4 == 0:
                self.print("Selected 4 points coordinates:")
                for i, coord in enumerate(self.coordinates[-4:]):
                    self.print(f"Point {i+1}: ({coord.x}, {coord.y})")

        cv2.destroyAllWindows()

        coord_output = np.float32([[c.x, c.y] for c in self.coordinates])

        # return self.coordinates
        return coord_output


if __name__ == "__main__":
    filepath = "./testfiles/extract_checker/CAL_OuterFrustum_v01_dW.exr"
    try:
        # Load the image using the imported load_exr function
        image = load_exr(filepath)

        # Create ImageProcessor instance with the loaded image
        processor = ColorcheckerExtractor(image)
        coords = processor.get_four_points()

        if len(coords) == 4:
            print("Obtained coordinates:", [(c.x, c.y) for c in coords])
        else:
            print("Failed to obtain 4 point coordinates.")
    except Exception as e:
        print(f"An error occurred: {e}")
