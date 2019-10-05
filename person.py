from enum import Enum
from typing import List


class BodyPart(Enum):
    NOSE = 'NOSE'
    LEFT_EYE = 'LEFT_EYE'
    RIGHT_EYE = 'RIGHT_EYE'
    LEFT_EAR = 'LEFT_EAR'
    RIGHT_EAR = 'RIGHT_EAR'
    LEFT_SHOULDER = 'LEFT_SHOULDER'
    RIGHT_SHOULDER = 'RIGHT_SHOULDER'
    LEFT_ELBOW = 'LEFT_ELBOW'
    RIGHT_ELBOW = 'RIGHT_ELBOW'
    LEFT_WRIST = 'LEFT_WRIST'
    RIGHT_WRIST = 'RIGHT_WRIST'
    LEFT_HIP = 'LEFT_HIP'
    RIGHT_HIP = 'RIGHT_HIP'
    LEFT_KNEE = 'LEFT_KNEE'
    RIGHT_KNEE = 'RIGHT_KNEE'
    LEFT_ANKLE = 'LEFT_ANKLE'
    RIGHT_ANKLE = 'RIGHT_ANKLE'


class Position:
    def __init__(self, x: int, y: int):
        self.x = x
        self. y = y


class KeyPoint:
    def __init__(self, body_part: BodyPart, position: Position, score: float):
        self.body_part = body_part
        self.position = position
        self.score = score


class Person:
    def __init__(self, key_points: List[KeyPoint], score: float):
        self.key_points = key_points
        self.score = score



