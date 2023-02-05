import os
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import imagehash


def download_image_from_message_to_cache(bot, message, image_folder):
    downloaded_file = bot.download_file(bot.get_file(message.photo[-1].file_id).file_path)
    cache_path = os.path.join(image_folder, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg")
    with open(cache_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    return cache_path


def reset_game_state(game_state_path):
    with open(game_state_path, "w") as f:
        f.write("my_cards: {}\n")


def get_game_state_path(bot, message, game_state_folder):
    game_state_path = os.path.join(game_state_folder, f"{message.from_user.username}.yaml")
    if not os.path.exists(game_state_path):
        reset_game_state(game_state_path)
    return game_state_path


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right poi
    # nt will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def detect_cards_on_monobg(image, target_sz=(None, None), cfg={}):
    """
    WARNING: image is expected to be of size around (1280, 960)
    otherwise, all the parameters would need to be tuned :)
    """
    blur = cv2.bilateralFilter(image, 17, 75, 45)
    gray_blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold will reveal the areas of highest contrast
    thresh = cv2.adaptiveThreshold(
        gray_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,25,2)

    # Detect edges, perform morphology operations to smooth out the
    # edges and fill potential small holes.
    edges = cv2.Canny(thresh, 10, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Probabilistic hough transform parameters:
    #   rho: The resolution of the parameter r in pixels. We use 1 pixel.
    #   theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    #   threshold: The minimum number of intersections to "*detect*" a line
    #   lines: A vector that will store the parameters (xstart,ystart,xend,yend)
    #       of the detected lines
    #   minLineLength: The minimum number of points that can form a line. Lines with
    #       less than this number of points are disregarded.
    #   maxLineGap: The maximum gap between two points to be considered in the same line.
    linesP = cv2.HoughLinesP(closed, 1, 1 * np.pi / 180, 1, None, 10, 10)
    hough_disp = np.zeros_like(closed)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(hough_disp, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    closed_hough = cv2.morphologyEx(hough_disp, cv2.MORPH_CLOSE, kernel2)
    cnts, _ = cv2.findContours(
        closed_hough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_disp = image.copy()
    ratio = 1.0  # ratio between original image and the one we're working with

    # loop over the contours
    cropped_images = []
    detected_idx = 0
    for cnt in cnts:
        if cfg.get("enclosing", "minrect") == 'minrect':
            min_rect = cv2.minAreaRect(cnt)
            box_points = cv2.boxPoints(min_rect)
            peri = cv2.arcLength(box_points, True)
            if peri < cfg.get("min_perimeter", 1500) or peri >= cfg.get("max_perimeter", 3500):
                continue
            scaled_box = scale_contour(box_points, 0.9)
            orig_rect_vertices = np.int0(box_points)
            min_rect_vertices = np.int0(scaled_box)
            cv2.drawContours(img_disp, [orig_rect_vertices], -1, (255, 0, 0), 4)
            cv2.drawContours(img_disp, [min_rect_vertices], -1, (0, 0, 255), 4)

            # calculate moments and calculate the center point
            M = cv2.moments(min_rect_vertices)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(
                img_disp, str(detected_idx), (cX, cY), cv2.FONT_HERSHEY_DUPLEX, 
                3, (127, 255, 0), 5, cv2.LINE_AA)
            detected_idx += 1

            cropped_image = four_point_transform(image, min_rect_vertices.reshape(4, 2) * ratio)
            cropped_images.append(cv2.resize(cropped_image, target_sz))
        elif cfg.get("enclosing", "minrect") == 'polydp':
            peri = cv2.arcLength(cnt, True)
            if peri < cfg.get("min_perimeter", 1500) or peri >= cfg.get("max_perimeter", 3500):
                continue
            # approximate the contour
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            cv2.drawContours(img_disp, [cnt], -1, (255, 0, 0), 4)

            # if the approximated contour has four points, then assume that the
            # contour is a book -- a book is a rectangle and thus has four vertices
            cv2.drawContours(img_disp, [approx], -1, (0, 255, 0), 4)
            if len(approx) == 4:
                cv2.drawContours(img_disp, [approx], -1, (0, 0, 255), 4)
                cropped_image = four_point_transform(image, approx.reshape(4, 2) * ratio)
                cropped_images.append(cv2.resize(cropped_image, target_sz))

    return {
        'debug': {
            'edges': cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB),
            'closed': cv2.cvtColor(closed_hough, cv2.COLOR_GRAY2RGB),
            'thresh': cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB),
        },
        'disp_image': img_disp,
        'cropped_images': np.array(cropped_images),
    }


def get_cards_from_image(image_path, config):
    # Detect cards on the image
    detected_cards = detect_cards_on_monobg(
        image=cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB),
        target_sz=(400, 600), cfg=config.cards_detection,
    )
    prefix, ext = os.path.splitext(image_path)
    detected_cards_paths = []
    for i, card_image in enumerate(detected_cards['cropped_images']):
        detected_cards_paths.append(
            f"{prefix}_card-{i}-of-{len(detected_cards['cropped_images'])}{ext}")
        Image.fromarray(card_image).save(detected_cards_paths[-1])
    if len(detected_cards['cropped_images']) > 0:
        detection_collage = cv2.hconcat(detected_cards['cropped_images'])
        detection_collage_path = f"{prefix}_all-cards{ext}"
        Image.fromarray(detection_collage).save(detection_collage_path)
    else:
        detection_collage_path = None
    debug_collage = cv2.hconcat([
        detected_cards['debug']['thresh'],
        # detected_cards['debug']['edges'],
        detected_cards['debug']['closed'],
        detected_cards['disp_image'],
    ])
    debug_collage_path = f"{prefix}_all-debug{ext}"
    Image.fromarray(debug_collage).save(debug_collage_path)

    card_hashes = []
    for card_image in detected_cards['cropped_images']:
        phash = imagehash.phash(Image.fromarray(card_image))
        dhash = imagehash.dhash(Image.fromarray(card_image))
        card_hashes.append(str(phash) + str(dhash))
    return {
        'hashes': card_hashes,
        'cards': detected_cards['cropped_images'],
        'cards_paths': detected_cards_paths,
        'collage_path': detection_collage_path,
        'debug_img_path': debug_collage_path,
    }
