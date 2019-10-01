import tensorflow as tf
import numpy as np
import cv2
from pose_detector import BodyPart, KeyPoint, Position, Person
import os


def sigmoid(x):
    return 1/(1+np.exp(-x))


def load_image(f='./sample_images/test1.png'):
    return cv2.imread(f)


def scale_image(img):
    height, width, _ = img.shape
    height_scale = height / 257
    width_scale = width / 257

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # INTER_AREA interpolation is faster than INTER_CUBIC
    reshaped_img = cv2.resize(rgb_img, dsize=(257, 257),
                              interpolation=cv2.INTER_AREA)
    reshaped_img = reshaped_img.reshape(1, 257, 257, 3)

    mean = 128.0
    std = 128.0

    scaled_img = (reshaped_img - mean) / std
    scaled_img = scaled_img.astype("float32")
    return scaled_img, height_scale, width_scale


def detect_person(img, interpreter) -> Person:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    heatmaps = interpreter.get_tensor(output_details[0]['index'])
    # print(f"heatmaps shape: {heatmaps.shape}")
    offsets = interpreter.get_tensor(output_details[1]['index'])
    # print(f"offsets shape: {offsets.shape}")
    # displacements_fwd = interpreter.get_tensor(output_details[2]['index'])
    # print(f"displacements_fwd shape: {displacements_fwd.shape}")
    # displacements_bwd = interpreter.get_tensor(output_details[3]['index'])
    # print(f"displacements_bwd shape: {displacements_bwd.shape}")

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
    interpreter = tf.lite.Interpreter(model_path='posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')
    interpreter.allocate_tensors()
    return interpreter


def main():
    interpreter = load_model()
    images = os.listdir("./sample_images")
    for i, image in enumerate(images):
        if image == ".DS_Store":
            continue
        img = load_image(os.path.join("./sample_images", image))
        annotate_img(img, interpreter)
        cv2.imwrite(os.path.join("output", f'image_{i}_w_cirlces.jpg'), img)

if __name__ == "__main__":
    main()

