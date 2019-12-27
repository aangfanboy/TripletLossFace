import sys
sys.path.append("../")

import json
import cv2, os
import numpy as np
import tensorflow as tf
from make_better_dataset_for_deepfake.main_data_creator import FaceExtractor


class Engine:
	def load_image(self, path):
		image = tf.io.read_file(path)
		image = tf.image.decode_jpeg(image, channels=3)

		image = tf.image.resize(image, (self.input_shape[0], self.input_shape[1]), method="nearest")

		return image.numpy()

	def __init__(self, model_path: str):
		self.faceExtractor = FaceExtractor()

		self.model = tf.keras.models.load_model(model_path, {"ReLU": tf.keras.layers.ReLU})
		self.model.summary()
		self.i = 0

		self.input_shape = self.model.layers[0].input_shape[0][1:]
		self.mistaken = []

		self.json_path = os.path.join("../datasets", "dfdc_train_part_45/metadata.json")

		with open(self.json_path, 'rb') as f:
			self.json = json.loads(f.read())

	def go_for_image(self, faces, load_image_first: bool = False, detect_faces_first: bool = True):
		y_map = {0: "real", 1: "fake"}
		aaa1 = faces

		if load_image_first:
			try:
				faces = self.load_image(faces)
				if not detect_faces_first:
					faces = [faces]
			except:
				print("error")
				return

		if detect_faces_first:
			faces = self.faceExtractor.extract([faces])[0]
		
		for face in faces:
			try:
				face = tf.image.resize(face, (self.input_shape[0], self.input_shape[1]), method="nearest")
				aa = tf.nn.softmax(self.model(tf.expand_dims(tf.cast(face, tf.float32)/255., 0)))
				if np.argmax(aa) == 0:
					print(y_map[np.argmax(aa)])
					print(aa[0][0]*100)
					print(aa[0][1]*100)
					print(aaa1)
					self.i += 1
					self.mistaken.append(aaa1)
					print("----------------------------------------")

				# cv2.imshow("face", face.numpy())
				# cv2.waitKey(0)
			except:
				print("error")
				continue

	def go_for_video(self, video_path, detect_faces_first: bool = True):
		y_map = {0: "REAL", 1: "FAKE"}
		all_frames = self.faceExtractor.extract_frames(video_path, 20)

		for face in all_frames:
			try:
				if detect_faces_first:
					faces_all, frames = self.faceExtractor.extract([face])
					for face in faces_all:
						face = face[0]

						face = tf.image.resize(face, (self.input_shape[0], self.input_shape[1]), method="nearest")
						aa = tf.nn.softmax(self.model(tf.expand_dims(tf.divide(tf.cast(face, tf.float32), 255.), 0)))

						if self.json[video_path.split("/")[-1]]["label"] == y_map[np.argmax(aa)]:  # self.json[video_path.split("/")[-1]]["label"]
							return True
						else:
							return False
			except:
				continue


if __name__ == '__main__':  # 216
	engine = Engine(model_path="models/softmax_deepfake_freezed.h5")
	from tqdm import tqdm

	q = tf.io.gfile.glob(os.path.join("../datasets", "dfdc_train_part_45/*.mp4"))
	all_num = len(q)
	bar = tqdm(all_num)
	trues = 0

	for n, path in enumerate(q):
		if engine.go_for_video(path, True):
			trues += 1

		bar.update(1)
		bar.set_description(f"{trues}/{(n+1)} = {(100*trues)/(n+1)}")

	print(engine.mistaken)
	print(engine.i)
