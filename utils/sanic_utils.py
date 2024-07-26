from decord import VideoReader,cpu
import os 
import numpy as np
import cv2 
import dlib
import face_recognition
from tqdm import tqdm

def real_or_fake(prediction):
    return {0: "REAL", 1: "FAKE"}[prediction ^ 1]


def real_or_fake_thres(probability, threshold=0.2):
    return "FAKE" if probability >= threshold else "REAL"


def face_rec(frames, p=None, klass=None):
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0
    mod = "cnn" if dlib.DLIB_USE_CUDA else "hog"

    for _, frame in tqdm(enumerate(frames), total=len(frames)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face_locations = face_recognition.face_locations(
            frame, number_of_times_to_upsample=0, model=mod
        )

        for face_location in face_locations:
            if count < len(frames):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                face_image = cv2.resize(
                    face_image, (224, 224), interpolation=cv2.INTER_AREA
                )
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                temp_face[count] = face_image
                count += 1
            else:
                break

    return ([], 0) if count == 0 else (temp_face[:count], count)


def extract_frames(video_file, frames_nums=15):
    vr = VideoReader(video_file, ctx=cpu(0))
    # Calculate the step size between frames
    step_size = max(1, len(vr) // frames_nums)
    return vr.get_batch(
        list(range(0, len(vr), step_size))[:frames_nums]
    ).asnumpy()  # seek frames with step_size

def is_video(vid):
    return os.path.isfile(vid) and vid.endswith(
        tuple([".avi", ".mp4", ".mpg", ".mpeg", ".mov"])
    )

def is_image(img):
    return os.path.isfile(img) and img.endswith(
        tuple([".jpg", ".jpeg", ".png", ".webp"])
    )

def set_result():
    return {
        "video": {
            "name": [],
            "pred": [],
            "klass": [],
            "pred_label": [],
            "correct_label": [],
        }
    }

def store_result(
    result, filename, y, y_val, klass, correct_label=None, compression=None
):
    result["video"]["name"].append(filename)
    result["video"]["pred"].append(y_val)
    result["video"]["klass"].append(klass.lower())
    result["video"]["pred_label"].append(real_or_fake(y))

    if correct_label is not None:
        result["video"]["correct_label"].append(correct_label)

    if compression is not None:
        result["video"]["compression"].append(compression)

    return result