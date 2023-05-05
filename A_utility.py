from calendar import c
import cv2
import numpy as np
import setting
import filterpy.kalman

FUNCTIONS = [
    "get_processed_frame",
    "mask_marker",
    "marker_center",
    "inpaint",
    "difference",
    "get_all_contour",
    "get_convex_hull_area",
    "draw_flow",
]
CLASS = ["ContactArea"]


def get_processed_frame(cam: cv2.VideoCapture):
    ret, frame = cam.read()
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    downsampled = cv2.pyrDown(rotated_frame).astype(np.uint8)
    return downsampled


def mask_marker(frame, debug=False):
    m, n = frame.shape[1], frame.shape[0]
    frame = cv2.pyrDown(frame).astype(
        np.float32
    )  # Down sampling, reduce the effect of noise

    blur = cv2.GaussianBlur(frame, (25, 25), 0)
    # Blur make the feature more obvious
    blur2 = cv2.GaussianBlur(frame, (15, 15), 0)
    diff = blur - blur2

    diff *= 20  # Arbitrary value

    diff[diff < 0.0] = 0.0
    diff[diff > 255.0] = 255.0

    # diff = cv2.GaussianBlur(diff, (5, 5), 0)

    THRESHOLD = 120
    mask_b = diff[:, :, 0] > THRESHOLD
    mask_g = diff[:, :, 1] > THRESHOLD
    mask_r = diff[:, :, 2] > THRESHOLD

    mask = (mask_b * mask_g + mask_b * mask_r + mask_g * mask_r) > 0

    if debug:
        cv2.imshow("maskdiff", diff.astype(np.uint8))
        cv2.imshow("mask", mask.astype(np.uint8) * 255)

    mask = cv2.resize(mask.astype(np.uint8), (m, n))

    return mask * 255  # Dot is white
    # return (1 - mask) * 255       # Dot is black


def marker_center(frame, debug=False):
    areaThresh1 = 20
    areaThresh2 = 500
    centers = []

    mask = mask_marker(frame, debug=debug)
    contours = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours[0]) < 5:  # if too little markers, then give up
        print("Too less markers detected: ", len(contours))
        return centers

    for i, contour in enumerate(contours[0]):
        x, y, w, h = cv2.boundingRect(contour)
        AreaCount = cv2.contourArea(contour)
        if (
            AreaCount > areaThresh1
            and AreaCount < areaThresh2
            and abs(np.max([w, h]) * 1.0 / np.min([w, h]) - 1) < 1
        ):
            t = cv2.moments(contour)
            mc = [t["m10"] / t["m00"], t["m01"] / t["m00"]]
            centers.append(mc)
            # cv2.circle(frame, (int(mc[0]), int(mc[1])), 10, (0, 0, 255), 2, 6)
    return centers


def inpaint(frame):
    mask = mask_marker(frame)
    frame_marker_removed = cv2.inpaint(
        frame, mask, 7, cv2.INPAINT_TELEA
    )  # Inpain the white area in the mask (aka the marker). The number is the pixel neighborhood radius

    return frame_marker_removed


def difference(frame, frame0, debug=False):
    # diff = cv2.absdiff(frame, frame0)

    # diff = frame - frame0

    diff = (frame * 1.0 - frame0) / 255.0 + 0.5  # Diff in range 0,1

    diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5
    diff_uint8 = (diff * 255).astype(np.uint8)
    diff_uint8_before = diff_uint8.copy()

    diff_uint8[diff_uint8 > 140] = 255
    diff_uint8[diff_uint8 <= 140] = 0

    diff_gray = cv2.cvtColor(diff_uint8, cv2.COLOR_BGR2GRAY)

    _, diff_thresh = cv2.threshold(
        diff_gray, 50, 255, cv2.THRESH_BINARY
    )  # Return 2 values, the second is the thresholded image
    diff_thresh_erode = cv2.erode(diff_thresh, np.ones((5, 5), np.uint8), iterations=2)

    diff_thresh_dilate = cv2.dilate(
        diff_thresh_erode, np.ones((5, 5), np.uint8), iterations=1
    )

    if debug:
        cv2.imshow("diff_uint8", diff_uint8_before)
        cv2.imshow("diff_uint8 after", diff_uint8)
        cv2.imshow("diff", diff.astype(np.uint8))
        cv2.imshow("diff_gray", diff_gray)
        cv2.imshow("diff_thresh_dilate", diff_thresh_dilate)
        cv2.imshow("diff_thresh", diff_thresh)
        cv2.imshow("diff_thresh_erode", diff_thresh_erode)

    return diff_thresh_dilate


def get_all_contour(diff_thresh_dilate, frame, debug=False):
    contours, hierarchy = cv2.findContours(
        diff_thresh_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    try:
        # for i, contour in enumerate(contours):
        #     area = cv2.contourArea(contour)
        #     if area < min_area:
        #         contours.pop(i)x
        # largest_contour = max(contours, key=cv2.contourArea)
        # ellipse = cv2.fitEllipse(largest_contour)

        merged_contour = np.concatenate(contours)
        # print("Merged contour: ", merged_contour.shape)
        ellipse = cv2.fitEllipse(merged_contour)

        img_ellipse = frame.copy()
        contour_ellipse = cv2.cvtColor(diff_thresh_dilate.copy(), cv2.COLOR_GRAY2BGR)

        cv2.ellipse(img_ellipse, ellipse, (0, 255, 0), 2)
        cv2.ellipse(contour_ellipse, ellipse, (0, 255, 0), 2)

        # Display the image with the ellipse
        if debug:
            cv2.imshow("Ellipse", img_ellipse)
            cv2.imshow("Ellipse on Contour", contour_ellipse)
    except:
        # print("No Contact found")
        pass
    return contours


def regress_line(all_points, frame, debug=False):
    line_frame = frame  # Draw on top of the original frame
    vx, vy, x, y = cv2.fitLine(all_points, cv2.DIST_L2, 0, 0.01, 0.01)

    slope = vy / vx

    lefty = int((-x * vy / vx) + y)
    righty = int(((line_frame.shape[1] - x) * vy / vx) + y)

    pt1 = (line_frame.shape[1] - 1, righty)
    pt2 = (0, lefty)

    midx = int((pt1[0] + pt2[0]) / 2)
    midy = int((pt1[1] + pt2[1]) / 2)

    # draw line on image
    cv2.line(line_frame, pt1, pt2, (0, 0, 255), 2)
    cv2.circle(line_frame, (int(midx), int(midy)), 10, (0, 0, 255), 2, 6)
    cv2.putText(
        line_frame,
        f"mid x: {midx}, mid y: {midy}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        line_frame,
        f"slope: {slope}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    if debug:
        cv2.imshow("Line", line_frame)
    return slope, (midx, midy)


def get_convex_hull_area(diff_thresh_dilate, frame, debug=False):
    contours, hierarchy = cv2.findContours(
        diff_thresh_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    img_hull = frame.copy()
    hull_area = 0
    slope = None
    center = None
    hull_mask = np.zeros(diff_thresh_dilate.shape, dtype=np.uint8)
    if len(contours) > 0:
        try:
            hull = []
            for i in range(len(contours)):  # Loop over contours
                for p in contours[i]:  # Loop over points in contours
                    hull.append(p[0])  # Append points to hull
            hull = np.array(hull)

            hullPoints = cv2.convexHull(
                hull, returnPoints=True
            )  # Return the hull vertices that enclose the points in hull

            cv2.drawContours(img_hull, [hullPoints], -1, (0, 255, 0), 2)
            hull_area = cv2.contourArea(hullPoints)
            slope, center = regress_line(
                hull, img_hull, debug=False
            )  # Draw the regressed line

            cv2.fillPoly(hull_mask, pts=[hullPoints], color=(255, 255, 255))
        except Exception as e:
            print("Hull", e)
            pass

    if debug:
        cv2.putText(
            img_hull,
            f"Hull Area: {hull_area}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Convex Hull", img_hull)
        cv2.imshow("Hull Mask", hull_mask)

    return hull_area, hull_mask, slope, center


def draw_flow(frame, flow):
    Ox, Oy, Cx, Cy, Occupied = flow
    K = 2
    drawn_frame = frame.copy()
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            # pt1 = (int(Ox[i][j]), int(Oy[i][j]))
            pt1 = (int(Cx[i][j]), int(Cy[i][j]))

            pt2 = (
                int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])),
                int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])),
            )
            color = (0, 255, 255)
            if Occupied[i][j] <= -1:
                color = (255, 255, 255)
            cv2.arrowedLine(drawn_frame, pt1, pt2, color, 2, tipLength=0.2)
    return drawn_frame


def draw_flow_mask(frame, flow, mask, debug=False):
    Ox, Oy, Cx, Cy, Occupied = flow
    K = 2
    drawn_frame = frame.copy()

    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=2)
    drawn_frame_and = cv2.bitwise_and(drawn_frame, drawn_frame, mask=mask)

    change = [0, 0]
    counter = 0
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            # pt1 = (int(Ox[i][j]), int(Oy[i][j]))

            if mask[int(Cy[i][j]), int(Cx[i][j])] == 255:
                dx = int(Cx[i][j] - Ox[i][j])
                dy = int(Cy[i][j] - Oy[i][j])
                pt1 = (int(Cx[i][j]), int(Cy[i][j]))

                pt2 = (
                    int(Cx[i][j] + K * dx),
                    int(Cy[i][j] + K * dy),
                )
                counter += 1
                change[0] += dx
                change[1] += dy
                color = (0, 255, 255)
                if Occupied[i][j] <= -1:
                    color = (255, 255, 255)

                cv2.arrowedLine(drawn_frame_and, pt1, pt2, color, 2, tipLength=0.2)
    if counter > 0:
        change[0] /= counter
        change[1] /= counter
        cv2.putText(
            drawn_frame_and,
            f"Average: {change}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    if debug:
        cv2.imshow("Flow Hull", mask)
    return drawn_frame_and, change


"""
This is from pytouch library 
"""


class ContactArea:
    def __init__(
        self, base=None, draw_poly=True, contour_threshold=100, *args, **kwargs
    ):
        self.base = base
        self.draw_poly = draw_poly
        self.contour_threshold = contour_threshold

    def __call__(self, target, base=None):
        base = self.base if base is None else base
        if base is None:
            raise AssertionError("A base sample must be specified for Pose.")
        diff = self._diff(target, base)
        diff = self._smooth(diff)
        contours = self._contours(diff)
        (
            poly,
            major_axis,
            major_axis_end,
            minor_axis,
            minor_axis_end,
        ) = self._compute_contact_area(contours, self.contour_threshold)
        if self.draw_poly:
            try:
                self._draw_major_minor(
                    target, poly, major_axis, major_axis_end, minor_axis, minor_axis_end
                )
                print("Drawn")
            except Exception as e:
                print("Error drawing major/minor axis: ", e)
                pass
        return (major_axis, major_axis_end), (minor_axis, minor_axis_end)

    def _diff(self, target, base):
        diff = (target * 1.0 - base) / 255.0 + 0.5

        cv2.imshow("diff1", diff)
        # print("Diff range", np.min(diff), np.max(diff))

        diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5
        cv2.imshow("diff2", diff)
        diff_abs = np.mean(np.abs(diff - 0.5), axis=-1)
        cv2.imshow("Diff_Abs", diff_abs)

        return diff_abs

    def _smooth(self, target):
        kernel = np.ones((64, 64), np.float32)
        kernel /= kernel.sum()
        diff_blur = cv2.filter2D(target, -1, kernel)
        cv2.imshow("Diff_Blur", diff_blur)
        return diff_blur

    def _contours(self, target):
        mask = ((np.abs(target) > 0.08) * 255).astype(np.uint8)  # Default > 0.04
        kernel = np.ones((16, 16), np.uint8)
        mask = cv2.erode(mask, kernel)

        cv2.imshow("Mask", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _draw_major_minor(
        self,
        target,
        poly,
        major_axis,
        major_axis_end,
        minor_axis,
        minor_axis_end,
        lineThickness=2,
    ):
        poly = None
        cv2.polylines(target, [poly], True, (255, 255, 255), lineThickness)
        cv2.line(
            target,
            (int(major_axis_end[0]), int(major_axis_end[1])),
            (int(major_axis[0]), int(major_axis[1])),
            (0, 0, 255),
            lineThickness,
        )
        cv2.line(
            target,
            (int(minor_axis_end[0]), int(minor_axis_end[1])),
            (int(minor_axis[0]), int(minor_axis[1])),
            (0, 255, 0),
            lineThickness,
        )

    def _compute_contact_area(self, contours, contour_threshold):
        poly = None
        major_axis = []
        major_axis_end = []
        minor_axis = []
        minor_axis_end = []

        for contour in contours:
            if len(contour) > contour_threshold:
                ellipse = cv2.fitEllipse(contour)
                poly = cv2.ellipse2Poly(
                    (int(ellipse[0][0]), int(ellipse[0][1])),
                    (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                    int(ellipse[2]),
                    0,
                    360,
                    5,
                )
                center = np.array([ellipse[0][0], ellipse[0][1]])
                a, b = (ellipse[1][0] / 2), (ellipse[1][1] / 2)
                theta = (ellipse[2] / 180.0) * np.pi
                major_axis = np.array(
                    [center[0] - b * np.sin(theta), center[1] + b * np.cos(theta)]
                )
                minor_axis = np.array(
                    [center[0] + a * np.cos(theta), center[1] + a * np.sin(theta)]
                )
                major_axis_end = 2 * center - major_axis
                minor_axis_end = 2 * center - minor_axis
        return poly, major_axis, major_axis_end, minor_axis, minor_axis_end
