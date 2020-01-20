import os
import time
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class MainData:
	@staticmethod
	def get_dataset_length(tf_dataset):
		return int(tf.data.experimental.cardinality(tf_dataset).numpy())

	def load_mnist_paths(self):
		paths = []
		labels = []

		if self.mnist_path is None:
			print("mnist_path cannot be None if you want to load mnist, you can set a mnist path to \"Main Data Class\" or " + 
				"you can set mnist_examples to \"True\" for \"run\" function.\n" + 
				"You have 5 seconds to stop the process and cancel running. Use Ctrl+C to do that."
			)

			time.sleep(5)

			return [], []

		print(f"Reading \"mnist\"...")

		for path in glob(self.mnist_path.rstrip("/")+"/" + "*/*.jpg"):
			class_name = path.split("/")[-2]

			paths.append(path)
			labels.append(int(class_name))

		print(f"Done reading \"mnist\".")		

		return paths, labels

	def create_triplet_dataset_for_deepfake(self, real_paths, real_labels, gen_paths, gen_labels, n_per_class: int = 25, data_path: str = "main_triplet.npy"):
		print("Starting to make dataset ready for triplet loss(deepfake), this may take a while.")
		try:
			new_x_data = np.load(data_path.replace(".npy", "_xdata.npy"), allow_pickle=True)
			new_y_data = np.load(data_path.replace(".npy", "_ydata.npy"), allow_pickle=True)
			print(f"Triplet dataset uploaded from {data_path}, it won't be re-creating based on your data.")
		except:
			real_paths, real_labels = np.array(real_paths), np.array(real_labels)
			gen_paths, gen_labels = np.array(gen_paths), np.array(gen_labels)
			new_x_data = []
			new_y_data = []
			unique_classes_gen = np.unique(gen_labels)

			bar = tqdm()
			iio = 0

			while iio < 1000:	
				iio += 1	
				main_type = np.random.choice([0, 1])
				gen_class = np.random.choice(unique_classes_gen, 1)

				if main_type == 0:
					main_indexes, neg_indexes = np.where(real_labels == gen_class)[0], np.where(gen_labels == gen_class)[0]
					min_n = min(min(len(main_indexes)*2, len(neg_indexes)), n_per_class)							

					main_indexes, neg_indexes = main_indexes[:min_n*2], neg_indexes[:min_n]
					main_images, negative_images = real_paths[main_indexes], gen_paths[neg_indexes]

					mi_s = int(len(main_indexes)/2)
					anchor_images, positive_images = main_images[mi_s:], main_images[:mi_s]

				else:
					main_indexes, neg_indexes = np.where(real_labels == gen_class)[0], np.where(gen_labels == gen_class)[0]
					min_n = min(min(len(main_indexes), len(neg_indexes)*2), n_per_class)							

					main_indexes, neg_indexes = main_indexes[:min_n], neg_indexes[:min_n*2]
					negative_images, main_images = real_paths[main_indexes], gen_paths[neg_indexes]

					mi_s = int(len(neg_indexes)/2)
					anchor_images, positive_images = main_images[mi_s:], main_images[:mi_s]

				for a, b, c in zip(anchor_images, positive_images, negative_images):
					new_x_data.append((a, b, c))
					new_y_data.append((gen_class, gen_class, main_type))

				real_paths = np.delete(real_paths, main_indexes)
				gen_paths = np.delete(gen_paths, neg_indexes)

				real_labels = np.delete(real_labels, main_indexes)
				gen_labels = np.delete(gen_labels, neg_indexes)
				unique_classes_gen = np.unique(gen_labels)

				bar.update(1)

				bar.set_description(f"{len(unique_classes_gen)} elements")

			new_x_data, new_y_data = np.array(new_x_data), np.array(new_y_data)
			bar.close()

			np.save(data_path.replace(".npy", "_xdata.npy"), new_x_data)
			np.save(data_path.replace(".npy", "_ydata.npy"), new_y_data)

		print(f"Dataset is ready for triplet loss(deepfake)! We have {len(new_x_data)} examples.")

		return new_x_data, new_y_data

	def create_main_triplet_dataset(self, paths, labels, n_per_class: int = 25, data_path: str = "main_triplet.npy"):
		print("Starting to make dataset ready for triplet loss, this may take a while base on the size of your dataset and \"n_per_class\"")
		try:
			new_x_data = np.load(data_path.replace(".npy", "_xdata.npy"))
			new_y_data = np.load(data_path.replace(".npy", "_ydata.npy"))
			print(f"Triplet dataset uploaded from {data_path}, it won't be re-creating based on your data.")
		except:
			paths, labels = np.array(paths), np.array(labels)
			new_x_data = []
			new_y_data = []
			unique_classes = np.unique(labels)

			bar = tqdm()

			while len(unique_classes) > 2:
				main_class, neg_class = 0, 0

				while main_class == neg_class:
					main_class, neg_class = np.random.choice(unique_classes, 2)

				main_indexes, neg_indexes = np.where(labels == main_class)[0], np.where(labels == neg_class)[0]
				min_n = min(min(len(main_indexes)*2, len(neg_indexes)), n_per_class)

				main_indexes, neg_indexes = main_indexes[:min_n*2], neg_indexes[:min_n]
				main_images, negative_images = paths[main_indexes], paths[neg_indexes]

				mi_s = int(len(main_indexes)/2)
				anchor_images, positive_images = main_images[mi_s:], main_images[:mi_s]

				for a, b, c in zip(anchor_images, positive_images, negative_images):
					new_x_data.append((a, b, c))
					new_y_data.append((main_class, main_class, neg_class))

				paths = np.delete(paths, main_indexes)
				paths = np.delete(paths, neg_indexes)

				labels = np.delete(labels, main_indexes)
				labels = np.delete(labels, neg_indexes)
				unique_classes = np.unique(labels)

				bar.update(1)

			new_x_data, new_y_data = np.array(new_x_data), np.array(new_y_data)
			bar.close()

			np.save(data_path.replace(".npy", "_xdata.npy"), new_x_data)
			np.save(data_path.replace(".npy", "_ydata.npy"), new_y_data)

		print(f"Dataset is ready for triplet loss! We have {len(new_x_data)} examples.")

		return new_x_data, new_y_data

	def __init__(self, main_path: str, mnist_path: str = None):
		self.main_path = main_path
		self.mnist_path = mnist_path

		self.real_paths, self.real_labels, self.real_y_map = None, None, None
		self.generated_paths, self.generated_labels, self.generated_y_map = None, None, None
		self.test_paths, self.test_labels = None, None

		print("Main Data Engine is ready.")

	def run(self, real_examples: bool = True, generated_examples: bool = True, test_examples: bool = True, mnist_examples: bool = False,
			real_examples_will_be_reading: list = ["105_classes_pins_dataset/", "CASIA_NEW_MAXPY/", "original_videos/"]):

		start_time = time.time()

		if real_examples:
			self.real_paths, self.real_labels, self.g_real_paths, self.g_real_labels, self.real_y_map = self.read_real_examples(
				will_be_reading=real_examples_will_be_reading
				)
		real_paths_time = time.time()

		if generated_examples:
			self.generated_paths, self.generated_labels, self.generated_y_map = self.read_generated_examples()
		generated_paths_time = time.time()

		if test_examples:
			self.test_paths, self.test_labels = self.read_test_data_as_real_or_generated()
		test_paths_time = time.time()

		if mnist_examples:
			self.mnist_paths, self.mnist_labels = self.load_mnist_paths()
			self.mnist_y_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
		mnist_paths_time = time.time()

		print(f"Asked data loaded in {round(time.time() - start_time, 3)} sc, Real took {round(real_paths_time - start_time, 3)} sc, " + 
			f"Generated took {round(generated_paths_time - real_paths_time, 3)} sc, Test took {round(test_paths_time - generated_paths_time, 3)} sc, " + 
			f"Mnist took {round(mnist_paths_time - test_paths_time, 3)} sc.")

		return True

	def create_tensorflow_dataset_object(self, paths, labels, test_rate: float = 0.1, test_data: tuple = (None, None), supportive: bool = False):
		print("Creating TensorFlow dataset object...")
		if type(test_data) != tuple:
			print("\"test_data\" must be tuple for 'create_tensorflow_dataset_object', test data will be taken from real data " + 
				f"with rate of {test_rate}. You have 5 seconds to stop the process and cancel running. Use Ctrl+C to do that.")

			time.sleep(5)


		paths_train, paths_test, labels_train, labels_test = train_test_split(paths, labels, test_size=test_rate, random_state=42)
		print("Dataset splitted by system, please make sure this is what you want.")

		dataset_train = tf.data.Dataset.from_tensor_slices((paths_train, labels_train)).shuffle(len(labels_train))
		dataset_test = tf.data.Dataset.from_tensor_slices((paths_test, labels_test)).shuffle(len(labels_test))
		print("TensorFlow dataset object created!")

		if not supportive:
			self.dataset_train = dataset_train
			self.dataset_test = dataset_test

		return dataset_train, dataset_test

	def read_real_examples(self, will_be_reading: list = ["105_classes_pins_dataset/", "CASIA_NEW_MAXPY/", "original_videos/"]):
		paths = []
		labels = []

		g_paths = []
		g_labels = []
		y_map = {}
		i_n = 0

		video_keys = []

		for dataset_name in will_be_reading:
			if dataset_name != "original_videos/":
				print(f"Reading \"{dataset_name}\"...")

				for path in glob(os.path.join(self.main_path, dataset_name) + "*/*.jpg"):
					class_name = path.split("/")[-2]

					if not class_name in y_map.keys():
						y_map[class_name] = i_n
						i_n += 1

					paths.append(path)
					labels.append(i_n)

				print(f"Done reading \"{dataset_name}\". have total {len(paths)}")

		if "original_videos/" in dataset_name:
			print(f"Reading \"preprocess/train/real/original_videos/\"...")
			for path in glob(os.path.join(self.main_path, "preprocess/train/real/original_videos/") + "*/*/*.jpg"):
				splitted = path.split("/")
				class_name = splitted[-3]
				video_key = splitted[-2]
				labeler = "_".join(video_key.split("_")[:-1])

				if not labeler in y_map.keys():
					y_map[labeler] = i_n
					i_n += 1

				# if not video_key in video_keys:
				# video_keys.append(video_key)

				g_paths.append(path)
				g_labels.append(y_map[labeler])

			print(f"Done reading \"preprocess/train/real/original_videos/\"...")

		print(f"Done reading real examples, {len(labels) + len(g_labels)} examples and {i_n} different classes found.")

		return paths, labels, g_paths, g_labels, y_map

	def read_generated_examples(self):
		paths = []
		labels = []
		y_map = {}
		i_n2 = 0

		video_keys = []
		b = None

		for method in ["method_A/", "method_B/"]:
			print(f"Reading \"preprocess/train/fake/{method}\"...")
			for path in glob(os.path.join(self.main_path, "preprocess/train/fake/", method) + "*/*/*/*.jpg"):  # IT WILL CHANGE TO train
				splitted = path.split("/")
				class_name = splitted[-3]
				video_key = splitted[-2]

				"""
				if not class_name in y_map.keys():
					y_map[class_name] = i_n2
					i_n2 += 1
				"""

				# if not video_key in video_keys:
					# video_keys.append(video_key)

				try:
					labels.append(self.real_y_map[class_name])
					paths.append(path)
				except KeyError:
					continue

			print(f"Done reading \"preprocess/train/fake/{method}\".")

		print(f"Done reading generated examples, {len(paths)} examples and {i_n2} different classes found.")

		return paths, labels, y_map

	def read_test_data_as_real_or_generated(self):  # 0 = fake, 1 = real
		paths = []
		labels = []

		video_keys = []

		for method in ["method_A/", "method_B/"]:
			print(f"Reading \"preprocess/test/fake/{method}\"...")
			for path in glob(os.path.join(self.main_path, "preprocess/test/fake/", method) + "*/*/*/*.jpg"):
				video_key = path.split("/")[-2]

				if not video_key in video_keys:
					video_keys.append(video_key)

					paths.append(path)
					labels.append(0)

			print(f"Done reading \"preprocess/test/fake/{method}\".")

		for method in ["original_videos/"]:
			print(f"Reading \"preprocess/test/real/{method}\"...")
			for path in glob(os.path.join(self.main_path, "preprocess/test/real/", method) + "*/*/*.jpg"):
				video_key = path.split("/")[-2]

				if not video_key in video_keys:
					video_keys.append(video_key)

					paths.append(path)
					labels.append(1)

			print(f"Done reading \"preprocess/test/real/{method}\".")

		print(f"Done reading test examples, {len(labels)} examples and {len(video_keys)} different video ids found.")

		if len(labels) != len(video_keys):
			print(f"Number of element in labels({len(labels)}) is not equal with video keys({len(video_keys)}). This is not suppose to happen." + 
				"You have 5 seconds to stop the process and cancel running. Use Ctrl+C to do that."
			)

			time.sleep(5)

		return paths, labels


if __name__ == '__main__':
	md = MainData("../datasets", mnist_path="../datasets/mnist")
	md.run(real_examples = True, generated_examples = False, test_examples = False, mnist_examples=False, real_examples_will_be_reading=[
	 "CASIA_NEW_MAXPY/", "105_classes_pins_dataset/"])

	real_new_x, real_new_y = md.create_main_triplet_dataset(md.real_paths, md.real_labels, 200, data_path="processed_data/casia_and_mine_triplet.npy")

	triplet_dataset = md.create_tensorflow_dataset_object(real_new_x, real_new_y, supportive=True)
	softmax_dataset_train, softmax_dataset_test = md.create_tensorflow_dataset_object(md.real_paths, md.real_labels, supportive=True)
