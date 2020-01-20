import os
import cv2
import dlib

from tqdm import tqdm
from main_data_engine import MainData


class Worker:
	def load_image(self, path):
		image = cv2.imread(path)

		return image

	def __init__(self, MDE: MainData):
		self.md = MDE

		self.detector = dlib.get_frontal_face_detector()
		self.sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

	def __call__(self):
		bar = tqdm(total=len(self.md.real_paths))
		for path, label in zip(self.md.real_paths, self.md.real_labels):
			image = self.load_image(path)

			dets = self.detector(image, 1)
			if len(dets) > 0:
				faces = dlib.full_object_detections()
				faces.append(self.sp(image, dets[0]))

				image = dlib.get_face_chips(image, faces)[0]

				p1 = path.replace("105_classes_pins_dataset", "105_classes_pins_dataset_aligned")
				os.makedirs(os.path.dirname(p1), exist_ok=True)
				cv2.imwrite(p1, image)

			bar.update(1)

		bar.close()

if __name__ == '__main__':
	md = MainData("../datasets", mnist_path="../datasets/mnist")
	md.run(real_examples = True, generated_examples = False, test_examples = False, mnist_examples=False, real_examples_will_be_reading=["105_classes_pins_dataset/"])

	print("DATA LOADED")
	worker = Worker(md)
	worker()

