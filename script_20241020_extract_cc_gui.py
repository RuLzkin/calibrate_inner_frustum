import cv2
from module_load_exr import load_exr


cc_coordinates = []
cc_img = None


def get_points(img):
    global cc_coordinates, cc_img
    cc_coordinates = []
    cc_img = img

    cv2.imshow("Input Image", cc_img)
    cv2.setMouseCallback("Input Image", click_event)

    while len(cc_coordinates) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cc_coordinates


def click_event(event, x, y, flags, param):
    global cc_coordinates, cc_img
    if event == cv2.EVENT_LBUTTONDOWN:
        cc_coordinates.append((x, y))
        cv2.circle(cc_img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Input Image", cc_img)

        if len(cc_coordinates) == 4:
            print("選択された4点の座標:")
            for i, coord in enumerate(cc_coordinates):
                print(f"Point {i+1}: {coord}")
            cv2.destroyAllWindows()


if __name__ == "__main__":
    filepath = "./testfiles/extract_checker/CAL_OuterFrustum_v01_dW.exr"
    img_test = load_exr(filepath)

    get_points(img_test)
