import cv2
import numpy as np
from module_load_exr import load_exr

# グローバル変数
coordinates = []
image = None
zoomed_image = None
zoom_factor = 4
is_zoomed = False
zoom_center = None


def add_instructions(img):
    h, w = img.shape[:2]
    instructions = [
        "Left click: Zoom / Select point, Right click: Cancel zoom",
        "R: Reset selection, ESC: Exit"
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

        # 黒い背景を追加
        cv2.rectangle(img, (x - 5, y - text_size[1] - 5), (w - padding + 5, y + 5), (0, 0, 0), -1)

        cv2.putText(img, instruction, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return img


def click_event(event, x, y, flags, param):
    global coordinates, image, zoomed_image, is_zoomed, zoom_center

    if event == cv2.EVENT_LBUTTONDOWN:
        if is_zoomed:
            original_x = int(zoom_center[0] + (x - image.shape[1] // 2) / zoom_factor)
            original_y = int(zoom_center[1] + (y - image.shape[0] // 2) / zoom_factor)
            coordinates.append((original_x, original_y))
            cv2.circle(image, (original_x, original_y), 5, (0, 255, 0), -1)
            is_zoomed = False
            display_image = add_instructions(image.copy())
            cv2.imshow('Image', display_image)
        else:
            zoom_and_display(x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if is_zoomed:
            is_zoomed = False
            display_image = add_instructions(image.copy())
            cv2.imshow('Image', display_image)


def zoom_and_display(center_x, center_y):
    global zoomed_image, is_zoomed, zoom_center

    half_width = image.shape[1] // (2 * zoom_factor)
    half_height = image.shape[0] // (2 * zoom_factor)

    x_start = max(0, center_x - half_width)
    y_start = max(0, center_y - half_height)
    x_end = min(image.shape[1], center_x + half_width)
    y_end = min(image.shape[0], center_y + half_height)

    zoomed_region = image[y_start:y_end, x_start:x_end]
    zoomed_image = cv2.resize(zoomed_region, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    display_image = add_instructions(zoomed_image.copy())
    cv2.imshow('Image', display_image)
    is_zoomed = True
    zoom_center = (center_x, center_y)


def get_four_points(img: np.ndarray):
    global coordinates, image, zoomed_image, is_zoomed
    coordinates = []
    image = img.copy()

    display_image = add_instructions(image.copy())
    cv2.imshow('Image', display_image)
    cv2.setMouseCallback('Image', click_event)

    while len(coordinates) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESCキーで終了
            break
        elif key == ord('r'):  # 'r'キーでリセット
            coordinates = []
            is_zoomed = False
            display_image = add_instructions(image.copy())
            cv2.imshow('Image', display_image)

        if len(coordinates) > 0 and len(coordinates) % 4 == 0:
            print("選択された4点の座標:")
            for i, coord in enumerate(coordinates[-4:]):
                print(f"Point {i+1}: {coord}")

    cv2.destroyAllWindows()
    return coordinates


if __name__ == "__main__":
    filepath = "./testfiles/extract_checker/CAL_OuterFrustum_v01_dW.exr"
    input_image = load_exr(filepath)

    if input_image is not None:
        coords = get_four_points(input_image)

        if len(coords) == 4:
            print("取得した座標:", coords)
        else:
            print("4点の座標を取得できませんでした。")
    else:
        print(f"エラー: 画像 '{filepath}' を読み込めませんでした。")
