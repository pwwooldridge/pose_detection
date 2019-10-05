import tensorflow as tf
import numpy as np
import cv2
from person import BodyPart, KeyPoint, Position, Person


def sigmoid(x):
    return 1/(1+np.exp(-x))


def load_image(f='./sample_images/test1.png'):
    return cv2.imread(f)


def standardize_image(img):
    mean = 128.0
    std = 128.0
    img_norm = (img - mean) / std
    img_norm = img_norm.astype("float32")
    return img_norm


def resize_image(img, dsize=(257,257)):
    # INTER_AREA interpolation is faster than INTER_CUBIC
    reshaped_img = cv2.resize(img, dsize=dsize,
                              interpolation=cv2.INTER_AREA)
    reshaped_img = reshaped_img.reshape(1, 257, 257, 3)
    return reshaped_img


def scale_image(img):
    height, width, _ = img.shape
    height_scale = height / 257
    width_scale = width / 257

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_image(img)
    img = standardize_image(img)
    return img, height_scale, width_scale


def detect_person(img, interpreter) -> Person:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    heatmaps = interpreter.get_tensor(output_details[0]['index'])
    offsets = interpreter.get_tensor(output_details[1]['index'])
    _, height, width, n_keypoints = heatmaps.shape

    keypoint_positions = {}
    for i in range(heatmaps.shape[-1]):
        heatmaps[0, ..., i] = sigmoid(heatmaps[0, ..., i])
        hm = heatmaps[0, ..., i]
        max_row, max_col = np.where(hm == np.max(hm))
        keypoint_positions[i] = (max_row[0], max_col[0])

    y_coords = {}
    x_coords = {}
    confidence_scores = {}

    for idx, v in keypoint_positions.items():
        position_y = v[0]
        position_x = v[1]
        y_coords[idx] = int((position_y / float(height-1)) * 257 + offsets[0][position_y][position_x][idx])
        x_coords[idx] = int((position_x / float(width-1)) * 257 + offsets[0][position_y][position_x][idx + n_keypoints])
        confidence_scores[idx] = heatmaps[0][position_y][position_x][idx]


    total_score = 0
    key_points = []
    for idx, b in enumerate(BodyPart):
        position = Position(x=x_coords[idx], y=y_coords[idx])
        key_points.append(KeyPoint(body_part=b, position=position, score=confidence_scores[idx]))
        total_score += confidence_scores[idx]

    p = Person(key_points, score=total_score/n_keypoints)
    return p


def annotate_img(img, interpreter):
    scaled_image, height_scale, width_scale = scale_image(img)
    person = detect_person(scaled_image, interpreter)
    for point in person.key_points:
        if point.score < 0.5:
            continue

        x = int(point.position.x * width_scale)
        y = int(point.position.y * height_scale)
        cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1,
                   lineType=8, shift=0)
    return img


def load_model():
    interpreter = tf.lite.Interpreter(model_path='./models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')
    interpreter.allocate_tensors()
    return interpreter

