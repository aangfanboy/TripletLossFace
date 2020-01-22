import tensorflow as tf
import sys
import json
import numpy as np
import cv2
from tqdm import tqdm

sys.path.append("../")

from deep_learning.make_better_dataset_for_deepfake.main_data_creator import FaceExtractor

class Claudia:
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

	def create_color_map(self):
		for key in self.json:
			try:
				self.colors[key]
			except KeyError:				
				color = tuple(np.random.choice(range(256), size=3))
				color = (int(color[0]), int(color[1]), int(color[2]))

				self.colors[key] = color

	def __init__(self, model_path: str):
		self.model = tf.keras.models.load_model(model_path)
		self.faceExtractor = FaceExtractor()

		self.input_shape = self.model.layers[0].input_shape[0][1:]

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

	def get_output_from_image(self, path, get_face: bool = True, l2: bool = False):
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

	def add_to_json(self, output):
		print("ADDED")
		i = len(self.json.keys()) + 1
		self.json[str(i)] = list(output.numpy().tolist())
		self.save_json()
		self.create_color_map()

		return str(i)

	def index_image(self, path, get_face: bool = True, print_out: bool = False, th: float = -0.60):
		image, output, all_frames = self.get_output_from_image(path, get_face, l2=True)

		mins = []
		assert len(all_frames) == len(output)

		for i in range(len(output)):
			founded = False

			for key in self.json:
				my_min = (100000, "")
				oo = self.json[key]
				dist = self.cosine_loss(tf.convert_to_tensor(oo), output[i]).numpy()

				if dist <= th:
					if dist < my_min[0]:
						my_min = (dist, key)

						founded = True

			if not founded:
				new_key = self.add_to_json(output[i])
				my_min = (-1., new_key)

			mins.append(my_min)

		return mins, image, all_frames

	def mark(self, image, min_im, all_frames):
		for (confidance, who), frame in zip(min_im, all_frames):
			try:
				color = self.colors[str(who)]
				x1, x2, y1, y2 = frame
				cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
				cv2.putText(image, f"id: {str(who)}- conf:{abs(round(float(confidance), 2))}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA) # -{round(float(confidance), 2)}
			except KeyError:
				continue

		return image


	def show_who_in_image(self, path, get_face: bool = True, show: bool = True, turn_rgb: bool = True):
		min_im, image, all_frames = self.index_image(path, get_face)

		for (confidance, who), frame in zip(min_im, all_frames):
			try:
				color = self.colors[str(who)]
				x1, x2, y1, y2 = frame
				cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
				cv2.putText(image, f"id: {str(who)}- conf:{abs(round(float(confidance), 2))}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA) # -{round(float(confidance), 2)}
			except KeyError:
				continue

		if turn_rgb:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		if show:
			cv2.imshow("a", image)
			cv2.waitKey(1)

		return image, min_im, all_frames

	def go_for_video(self, path, i: int = -1):
		cap = cv2.VideoCapture(path)
		n = 0

		if i != -1:
			bar = tqdm(total=i)
		else:
			bar = tqdm()

		min_im, all_frames = None, None
		while True:
			try:
				result, frame = cap.read()
				if n % 6 == 0:
					if self.video_writer is None:
						h, w, c = frame.shape
						self.video_writer = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*"MJPG"), 30,(w,h))

					if not result:
						break

					frame, min_im, all_frames = self.show_who_in_image(frame, True, False, turn_rgb=False)
					self.video_writer.write(frame)

				else:
					frame = self.mark(frame, min_im, all_frames)
					self.video_writer.write(frame)

				n += 1
				bar.update()
			except Exception as e:
				print(e)
				continue


		self.video_writer.release()



if __name__ == '__main__':
	claudia = Claudia("../deep_learning/models/triplet_inception_resnet_v1_0.h5")

	claudia.go_for_video("bbt_test1.mp4")