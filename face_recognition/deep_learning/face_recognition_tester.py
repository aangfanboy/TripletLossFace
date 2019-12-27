import sys
sys.path.append("../")

import json
import cv2, os
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from make_better_dataset_for_deepfake.main_data_creator import FaceExtractor


class Engine:
	def load_full(self, path):
		image = tf.io.read_file(path)
		image = tf.image.decode_jpeg(image, channels=3)

		return image.numpy()

	def set_image(self, image):
		image = tf.image.resize(image, (self.input_shape[0], self.input_shape[1]), method="nearest")

		return tf.cast(image, tf.float32)/255.

	def save_json(self):
		with open(self.json_path, 'w') as f:
			json.dump(self.json, f)

	def __init__(self, model_path: str):
		self.faceExtractor = FaceExtractor()

		self.model = tf.keras.models.load_model(model_path)
		self.model.summary()
		self.i = 0

		self.input_shape = self.model.layers[0].input_shape[0][1:]
		self.mistaken = []

		self.json_path = "my_data.json"

		try:
			with open(self.json_path, 'rb') as f:
				self.json = json.loads(f.read())
		except:
			with open(self.json_path, 'w+') as f:
				self.json = {}		

		self.colors = {}
		self.create_color_map()

		self.video_writer = None
		self.cosine_loss = tf.keras.losses.CosineSimilarity()


	def image_output(self, path, get_face: bool = True, l2: bool = False):
		if type(path) == str:
			image = self.load_full(path)
		else:
			image = path

		if get_face:
			faces, all_frames = self.faceExtractor.extract([image])
			faces = faces[0]
			all_frames = all_frames[0]

		else:
			faces = [image]
			all_frames = [(0,image.shape[1], 0, image.shape[0])]

		outputs = []
		for face in faces:
			face = self.set_image(face)
			output = self.model(tf.expand_dims(face, axis=0))

			if l2:
				output = tf.nn.l2_normalize(output, 1, 1e-10)

			outputs.append(output)

		return image, outputs, all_frames

	def detect_which(self, path, get_face: bool = True, print_out: bool = False):
		image, output, all_frames = self.image_output(path, get_face, l2=True)

		mins = []

		for i in range(len(output)):
			min_im = (10000, "")

			for key in self.json:
				oo = self.json[key]
				dist = abs(-1 - self.cosine_loss(tf.convert_to_tensor(oo), output[i]).numpy())

				if dist < min_im[0]:
					min_im = (dist, key)

				if print_out:
					print(f"Distance between this and {key} is --> {dist}")

			mins.append(min_im)

		return mins, image, all_frames

	def label_person(self, path, name, update_after: bool = False):
		image, output, all_frames = self.image_output(path, l2=True)
		output = output[0]
		self.json[name] = list(output.numpy()[0].tolist())

		if update_after:
			self.save_json()

	def create_color_map(self):
		for key in self.json:
			color = tuple(np.random.choice(range(256), size=3))
			color = (int(color[0]), int(color[1]), int(color[2]))

			self.colors[key] = color

	def show_who_in_image(self, path, get_face: bool = True, show: bool = True, turn_rgb: bool = True):
		min_im, image, all_frames = self.detect_which(path, get_face)

		for (confidance, who), frame in zip(min_im, all_frames):
			color = self.colors[who]
			x1, x2, y1, y2 = frame
			cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
			cv2.putText(image, f"{who}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA) # -{round(float(confidance), 2)}

		if turn_rgb:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		if show:
			cv2.imshow("a", image)
			cv2.waitKey(0)

		return image

	def go_for_video(self, path, i: int = -1):
		cap = cv2.VideoCapture(path)
		n = 0

		if i != -1:
			bar = tqdm(total=i)
		else:
			bar = tqdm()

		while True:
			try:
				result, frame = cap.read()
				if self.video_writer is None:
					h, w, c = frame.shape
					self.video_writer = cv2.VideoWriter('examples/witcher3/video.avi', cv2.VideoWriter_fourcc(*"MJPG"), 30,(w,h))

				if not result:
					break

				frame = self.show_who_in_image(frame, True, False, turn_rgb=False)
				self.video_writer.write(frame)
				# cv2.imshow("frame", frame)
				# cv2.waitKey(1)

				if i != -1 and n == i:
					break

				n += 1
				bar.update()

			except:
				continue

		self.video_writer.release()


if __name__ == '__main__':
	engine = Engine(model_path="models/triplet_inception_resnet_v1_0.h5")

	"""
	engine.label_person("examples/witcher3/marilka.png", "marilka", True)
	engine.label_person("examples/witcher3/cirilla.jpeg", "cirilla", True)
	engine.label_person("examples/witcher3/tissaia.jpg", "tissaia", True)
	engine.label_person("examples/witcher3/geralt.jpg", "geralt", True)
	engine.label_person("examples/witcher3/renfri.jpg", "renfri", True)
	engine.label_person("examples/witcher3/jaskier.jpeg", "jaskier", True)
	engine.label_person("examples/witcher3/yennefer.jpg", "yennefer", True)
	engine.label_person("examples/witcher3/stregobor.jpg", "stregobor", True)

	engine.label_person("examples/silicon_valley/dinesh.jpg", "dinesh", True)
	engine.label_person("examples/silicon_valley/gilfoyle.jpg", "gilfoyle", True)
	engine.label_person("examples/silicon_valley/jared.jpeg", "jared", True)
	engine.label_person("examples/silicon_valley/monica.jpg", "monica", True)
	engine.label_person("examples/silicon_valley/richard.png", "richard", True)

	engine.label_person("examples/random/ki1.jpg", "melis", True)
	engine.label_person("examples/random/h1.jpg", "hannah", True)
	engine.label_person("examples/random/t1.jpg", "tugba", True)
	engine.label_person("examples/random/hi1.png", "hilal", True)
	
	engine.label_person("examples/bbt/amy.jpg", "amy", True)
	engine.label_person("examples/bbt/bernadette.jpg", "bernadette", True)
	engine.label_person("examples/bbt/howard.jpg", "howard", True)
	engine.label_person("examples/bbt/leonard.jpg", "leonard", True)
	engine.label_person("examples/bbt/penny.jpg", "penny", True)
	engine.label_person("examples/bbt/rajesh.jpg", "rajesh", True)
	engine.label_person("examples/bbt/sheldon.jpg", "sheldon", True)

	"""

	engine.create_color_map()

	for path in tf.io.gfile.glob("examples/silicon_valley/allatonce*.*"):
		try:
			engine.show_who_in_image(path, True)
		except Exception as e:
			print(f"error on {path}")
			print(e)
			continue