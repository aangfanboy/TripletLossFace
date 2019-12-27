import os
import cv2
import json

from tqdm import tqdm

import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN

def extract_frames(path, number_of_frames: int = -1):
    i = 0
    frame_list = []

    cap = cv2.VideoCapture(path)
    while True:
        if number_of_frames != -1 and i == number_of_frames:
            break

        result, frame = cap.read()
        if not result:
            break

        frame_list.append(frame)

        i += 1

    return frame_list

class FaceExtractor:
    @staticmethod
    def extract_frames(path, number_of_frames: int = -1):
        i = 0
        frame_list = []

        cap = cv2.VideoCapture(path)
        while True:
            if number_of_frames != -1 and i == number_of_frames:
                break

            result, frame = cap.read()
            if not result:
                break

            frame_list.append(frame)

            i += 1

        return frame_list

    def __init__(self):
        self.detector = MTCNN()

    def __get_scale___(self, w, h):
        if w == 3840:
            scale = 24
        elif w >= 1072:
            scale = 8
        elif w == 720:
            scale = 4
        elif w == 320 and h == 180:
            scale = 1
        elif w >= 270:
            scale = 2
        else:
            scale = 1
        
        return scale

    def detect(self, frame, w_i: float = 20., h_i: float = 10., confidence_th: float = 0.95):
        w, h = frame.shape[0], frame.shape[1]
        faces = []
        frames_all = []
        
        results = self.detector.detect_faces(frame)
        for i, _ in enumerate(results):
            if results[i]["confidence"] > confidence_th:
                x1, y1, width, height = [x for x in results[i]['box']]
                x2, y2 = x1 + width, y1 + height

                face = frame[y1:y2, x1:x2]
                hh1, ww1 = int(int(y2-y1)*h_i/100), int(int(x2-x1)*w_i/100)
                faces.append(frame[y1-hh1:y2+hh1, x1-ww1:x2+ww1])
                frames_all.append((x1-ww1, x2+ww1, y1-hh1, y2+hh1))

        return faces, frames_all
            
    def extract(self, frames):
        all_faces = []
        all_frames = []

        for frame in frames:
            q, qq = self.detect(frame)
            if q is not None:
                all_faces.append(q)
                all_frames.append(qq)

        return all_faces, all_frames


def generate_dataset(input_dir, output_dir):
    face_extractor = FaceExtractor()

    with open(os.path.join(input_dir, 'dataset.json'), 'rb') as f:
        dataset = json.loads(f.read())

    for video_path, data in tqdm(dataset.items()):
        set_name = data['set']
        label = data['label']

        if set_name == 'test':
            continue

        frame_list = extract_frames(os.path.join(input_dir, video_path))

        face_list, _ = face_extractor.extract(frame_list)

        del frame_list

        output_path = os.path.join(output_dir, set_name, label, video_path)
        os.makedirs(output_path, exist_ok=True)

        for frame_num, faces in enumerate(face_list):
            if faces is not None:
                for i, face in enumerate(faces):
                    #cv2.imshow("a", face)
                    #cv2.waitKey(0)
                    output_file = "{}/{}_{}.jpg".format(output_path, frame_num, i)
                    cv2.imwrite(output_file, face)

            else:
            	print("hata")


if __name__ == '__main__':
    generate_dataset("dfdc_train_part_45", "deneme")