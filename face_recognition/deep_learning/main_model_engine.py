import os
import cv2
import numpy as np
import tensorflow as tf

from main_data_engine import MainData
from model_archs import inception_resnet_v1


class TensorBoardCallback:
	def initialize(self):
		self.file_writer = tf.summary.create_file_writer(logdir=self.logdir)
		print(f"TensorBoard Callback initialized to {self.logdir}.")

	def __init__(self, logdir: str = "tensorboard_graphs/"):
		self.logdir = logdir
		self.file_writer = None

		self.initial_step = 0

		print("TensorBoard Callback created.")

	def __call__(self, data_json: dict, description: str = None):
		with self.file_writer.as_default():
			for key in data_json:
				tf.summary.scalar(key, data_json[key], step=self.initial_step, description=description)

		self.initial_step += 1

	def add_text(self, name: str, data: str, step: int):
		with self.file_writer.as_default():
			tf.summary.text(name, data, step=step)


class MainModel:
	def calculate_accuracy(self, y_real, y_pred):
		return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y_pred), -1), tf.argmax(y_real, -1)), tf.float32))

	def triplet_loss(self, y_real, output):
		output = tf.nn.l2_normalize(output, 1, 1e-10)
		anchor, positive, negative = tf.unstack(tf.reshape(output, (-1, 3, self.n_features)), num=3, axis=1)

		positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
		negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

		loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), 0.2)
		loss = tf.reduce_mean(tf.maximum(loss_1, 0.0), 0)

		return loss

	@tf.function
	def train_step(self, x, y):
		with tf.GradientTape() as tape:
			output = self.model(x, training=True)
			if self.use_center_loss:
				features, output = output
				center_loss = self.center_loss(features, y, 0.95)
				loss = self._loss_function(y, output)

				loss = (center_loss*self.center_lambda) + loss

			else:
				loss = self._loss_function(y, output)

		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		return loss, output

	@tf.function
	def test_step(self, x, y):
		with tf.GradientTape() as tape:
			output = self.model(x, training=False)
			if self.use_center_loss:
				features, output = output
				center_loss = self.center_loss(features, y, 0.95)
				loss = self._loss_function(y, output)

				loss = (center_loss*self.center_lambda) + loss

			else:
				loss = self._loss_function(y, output)

		return loss, output

	def center_loss(self, features, label, alfa):
		label = tf.reshape(tf.argmax(label, -1), [-1])

		centers_batch = tf.gather(self.centers, label)
		diff = (1 - alfa) * (centers_batch - features)
		self.centers = tf.compat.v1.scatter_sub(self.centers, label, diff)
		with tf.control_dependencies([self.centers]):
			loss = tf.reduce_mean(tf.square(features - centers_batch))

		return loss

	@tf.function
	def lossles_test(self, x):
		output = self.model(x, training=False)

		return output

	def load_image(self, path):
		image = tf.io.read_file(path)
		image = tf.image.decode_jpeg(image, channels=3)
		image = tf.image.resize(image, (self.input_shape[0], self.input_shape[1]), method="nearest")
		image = tf.image.random_flip_left_right(image)

		return tf.divide(tf.cast(image, tf.float32), 255.)

	def mapper_triplet(self, path, label):
		return [self.load_image(path[0]), self.load_image(path[1]), self.load_image(path[2])], label

	def mapper_softmax(self, path, label):
		return self.load_image(path), tf.one_hot(label, self.reverse_y_map_length)

	def set_dataset_ready(self, dataset, set_map: bool = True, set_batch: bool = True):
		dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

		if set_map:
			try:
				if self.mode == "triplet":
					dataset = dataset.map(self.mapper_triplet, tf.data.experimental.AUTOTUNE)
				elif self.mode == "softmax" or self.mode == "sparse softmax":
					dataset = dataset.map(self.mapper_softmax, tf.data.experimental.AUTOTUNE)
				else:
					raise Exception(f"There is no mapping function for {self.mode}, please fix.")
			except ValueError:
				raise Exception(f"You must set dataset for {self.mode} if you want to use.")

		if set_batch:
			dataset = dataset.batch(self.batch_size)

		print("Dataset ready for stream.")

		return dataset

	def __init__(self, mode: str = "softmax", use_center_loss: bool = False, selected_loss=None, y_map=None):
		self.mode = mode
		self._loss_function = None
		self.use_center_loss = use_center_loss

		if y_map is None:
			y_map = self.data_engine.real_y_map

		self.reverse_y_map = {v: k for k, v in y_map.items()}
		self.reverse_y_map_length = len(self.reverse_y_map)
		self.y_map = y_map

		self.center_lambda = 0.5

		if self.use_center_loss:
			if os.path.exists(f"models/centers_for_{self.mode}_{self.name}.npy"):
				self.centers = tf.Variable(np.load(f"models/centers_for_{self.mode}_{self.name}.npy"), trainable=False)
			else:
				init = tf.constant_initializer(0.)
				self.centers = tf.Variable(init([self.reverse_y_map_length, self.n_features]), trainable=False)

		if selected_loss is None:
			if self.mode == "triplet":
				self._loss_function = self.triplet_loss

			if self.mode == "softmax":
				self._loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

			if self.mode == "sparse softmax":
				self._loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

			if self._loss_function is None:
				raise Exception(f"ERROR, {self.mode} is not an option. Use: 'triplet', 'softmax' or 'sparse softmax' ")
		if selected_loss is not None:
			self._loss_function = selected_loss

		self.dataset_train = self.set_dataset_ready(self.data_engine.dataset_train)
		self.dataset_test = self.set_dataset_ready(self.data_engine.dataset_test)

		if self.generator_dataset_train is not None and self.generator_dataset_test is not None:
			self.generator_dataset_train = self.set_dataset_ready(self.generator_dataset_train)
			self.generator_dataset_test = self.set_dataset_ready(self.generator_dataset_test)

		if self.mode == "triplet":
			self.data_test = self.load_test_data()

	def load_test_data(self, n: int = 1):
		data_test = {"mine1": None, "mine2": None, "mine3": None, "gan1": None, "gan2": None}

		for type_x in os.listdir("test_folder"):

			if os.path.exists(os.path.join("test_folder", type_x, "1.jpg")):
				path1 = os.path.join("test_folder", type_x, "1.jpg")
				path2 = os.path.join("test_folder", type_x, "2.jpg")
				path3 = os.path.join("test_folder", type_x, "3.jpg")

			else:
				path1 = os.path.join("test_folder", type_x, "1.jpeg")
				path2 = os.path.join("test_folder", type_x, "2.jpeg")
				path3 = os.path.join("test_folder", type_x, "3.jpeg")

			if "gan" in type_x:
				if data_test["gan1"] is None:
					data_test["gan1"] = [self.load_image(path1), self.load_image(path2), self.load_image(path3)]

				elif data_test["gan2"] is None:
					data_test["gan2"] = [self.load_image(path1), self.load_image(path2), self.load_image(path3)]

			if "mine" in type_x:
				if data_test["mine1"] is None:
					data_test["mine1"] = [self.load_image(path1), self.load_image(path2), self.load_image(path3)]

				elif data_test["mine2"] is None:
					data_test["mine2"] = [self.load_image(path1), self.load_image(path2), self.load_image(path3)]

				elif data_test["mine3"] is None:
					data_test["mine3"] = [self.load_image(path1), self.load_image(path2), self.load_image(path3)]

		return data_test

	def test_with_created_data(self, epoch: int):
		for key in self.data_test:
			x = self.data_test[key]
			x = tf.reshape(x, (-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))

			output = self.lossles_test(x) 
			output = tf.nn.l2_normalize(output, 1, 1e-10)

			anchor, positive, negative = tf.unstack(tf.reshape(output, (-1, 3, self.n_features)), num=3, axis=1)

			d1 = round(float(np.linalg.norm(anchor - positive)), 3)  # small
			d2 = round(float(np.linalg.norm(anchor - negative)), 3)  # big
			d3 = round(float(np.linalg.norm(negative - positive)), 3)  # big

			self.tensorboard.add_text(str(key), f"A-P: {d1} || A-N: {d2} || N-P: {d3}", step=epoch)

	def train_loop(self, n: int = 1000, use_accuracy: bool = False):
		print(f"Training loop is activated.")
		
		self.tensorboard.initialize()
		data_json = {"loss": None}
		if use_accuracy:
			data_json["acc"] = None

		for epoch in range(self.epochs):
			print("\n\n")
			bar = tf.keras.utils.Progbar(
				self.data_engine.get_dataset_length(self.dataset_train) + 
				self.data_engine.get_dataset_length(self.dataset_test),
				stateful_metrics=["loss", "acc"]
				)

			loss_mean = tf.keras.metrics.Mean()
			acc_mean = tf.keras.metrics.Mean()

			for x, y in self.dataset_train:
				x = tf.reshape(x, (-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))

				loss, output = self.train_step(x, y) 
				loss_mean(loss)

				data_json["loss"] = loss
				main_bar_list = [["loss", loss_mean.result().numpy()]]

				if use_accuracy:
					acc = self.calculate_accuracy(y, output)
					acc_mean(acc)
					data_json["acc"] = acc
					main_bar_list.append(["acc", acc_mean.result().numpy()])

				self.tensorboard(data_json=data_json)

				bar.add(1, main_bar_list)

				if bar._seen_so_far % n == 0:
					self.model.save(self.new_name)
					if self.use_center_loss:
						np.save(f"models/centers_for_{self.mode}_{self.name}.npy", self.centers.numpy())

					if self.mode == "triplet":
						self.test_with_created_data(epoch)

					loss_mean.reset_states()
					acc_mean.reset_states()

			for x, y in self.dataset_test:
				x = tf.reshape(x, (-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
				loss, output = self.test_step(x, y) 

				main_bar_list = [["validation loss", loss]]
				if use_accuracy:
					acc = self.calculate_accuracy(y, output)
					main_bar_list.append(["validation acc", acc])

				bar.add(1, main_bar_list)


			if self.mode == "triplet":
				self.test_with_created_data(epoch)

			self.model.save(self.new_name)
			if self.use_center_loss:
				np.save(f"models/centers_for_{self.mode}_{self.name}.npy", self.centers.numpy())

	def test_without_monitoring(self,  dataset=None, use_accuracy: bool = False):
		if dataset is None:
			dataset = self.dataset_test

		bar = tf.keras.utils.Progbar(self.data_engine.get_dataset_length(dataset))

		for x, y in dataset:
			x = tf.reshape(x, (-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
			loss, output = self.test_step(x, y) 

			main_bar_list = [["validation loss", loss]]
			if use_accuracy:
				acc = self.calculate_accuracy(y, output)
				main_bar_list.append(["validation acc", acc])

			bar.add(1, main_bar_list)

	def test_with_monitoring(self, dataset=None):
		cosine_loss = tf.keras.losses.CosineSimilarity()
		if dataset is None:
			dataset = self.dataset_test

		if self.mode == "triplet":
			for x, y in dataset:
				x1 = tf.reshape(x, (-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))

				loss, output = self.test_step(x1, y) 
				output = tf.nn.l2_normalize(output, 1, 1e-10)

				anchor, positive, negative = tf.unstack(tf.reshape(output, (-1, 3, self.n_features)), num=3, axis=1)

				for i in range(anchor.shape[0]):
					d1 = round(float(np.linalg.norm(anchor[i] - positive[i])), 3)  # small
					d2 = round(float(np.linalg.norm(anchor[i] - negative[i])), 3)  # big
					d3 = round(float(np.linalg.norm(negative[i] - positive[i])), 3)  # big

					print("-------------")
					print(cosine_loss(anchor[i], positive[i]).numpy())
					print(cosine_loss(anchor[i], negative[i]).numpy())
					print(cosine_loss(negative[i], positive[i]).numpy())
					print()

					print(d1)
					print(d2)
					print(d3)
					print("-------------")

					cv2.imshow("a", x[i][0].numpy())
					cv2.imshow("b", x[i][1].numpy())
					cv2.imshow("c", x[i][2].numpy())

					cv2.waitKey(0)

		if self.mode == "softmax" or self.mode == "sparse softmax":
			for x, y in dataset:

				loss, output = self.test_step(x, y) 
				output = tf.argmax(tf.nn.softmax(output), -1).numpy().astype(int)

				y = tf.argmax(y, -1).numpy().astype(int)
				for i in range(x.shape[0]):
					try:
						print(f"Real Output --> {self.reverse_y_map[y[i]]}, Predicted Output --> {self.reverse_y_map[output[i]]}")
					except KeyError:
						print(f"[WARNING] Wrong Y-Map, please fix. Real Output ID --> {y[i]}, Predicted Output ID --> {output[i]}")

					cv2.imshow("face", x[i].numpy())
					cv2.waitKey(0)		 


class InceptionRV1(MainModel):
	name = "inception_resnet_v1"

	def __init__(self, data_engine, generator_dataset_train, generator_dataset_test, batch_size: int, epochs: int, lr: float,  use_center_loss: bool, mode: str = "softmax",
		selected_loss=None, y_map=None, input_shape: tuple = (128, 128, 3), kernel_regularizer=tf.keras.regularizers.l2(), model_id: int = 0,
		n_features: int = 128, pooling=tf.keras.layers.GlobalAveragePooling2D, bn_at_the_end: bool = False, new_name: str = None
		):

		self.model_path = f"models/{mode}_{self.name}_0.h5"
		self.data_engine = data_engine
		self.generator_dataset_train = generator_dataset_train
		self.generator_dataset_test = generator_dataset_test
		self.batch_size = batch_size
		self.epochs = epochs
		self.optimizer = tf.keras.optimizers.Adam(lr)
		self.n_features = n_features
		self.pooling = pooling
		self.input_shape = input_shape
		self.bn_at_the_end = bn_at_the_end
		self.kernel_regularizer = kernel_regularizer

		os.makedirs("models", exist_ok=True)

		self.tensorboard = TensorBoardCallback(logdir="graphs/")

		print(f"Activated Model --> {self.name}")

		if new_name is None:
			self.new_name = self.model_path
		else:
			self.new_name = new_name

		# self.new_name = f"models/{new_name}.h5"
		# self.model_path = f"models/{new_name}.h5"

		super().__init__(mode, use_center_loss, selected_loss, y_map)

	def make_model_ready(self, model, new_activation):
		for layer in model.layers:
			if self.kernel_regularizer is not None:
				layer.kernel_regularizer = self.kernel_regularizer

		return model

	def get_model(self, new_activation=tf.keras.layers.ReLU(), dropout_rate: float = 0.0, from_softmax=False, from_triplet=False, freeze=False):
		if freeze and not from_triplet:
			raise Exception("if you want to use 'freeze', you must set 'from_triplet' to True.")

		path = self.model_path
		if from_softmax:
			path = path.replace(self.mode, "softmax")
		if from_triplet:
			path = path.replace(self.mode, "triplet")

		if tf.io.gfile.exists(path):
			model = tf.keras.models.load_model(path, {"ReLU": tf.keras.layers.ReLU})  # {"LeakyReLU": tf.keras.layers.LeakyReLU}

			if self.mode == "triplet" and from_softmax:
				model = tf.keras.models.Model(model.layers[0].input, model.layers[-3].output) 
				print("BE CAREFUL, DENSE LAYER HAS BEEN REMOVED")

				if not self.bn_at_the_end and "BatchNormalization" in str(model.layers[-1]):
					model = tf.keras.models.Model(model.layers[0].input, model.layers[-2].output) 
					print("BE CAREFUL, BatchNormalization LAYER HAS BEEN REMOVED")					

				model.layers[-1].activation = None

			if (self.mode == "softmax" or self.mode == "sparse softmax") and from_triplet:
				model.layers[-1].activation = tf.keras.layers.ReLU()
				x = model.layers[-1].output
				if self.bn_at_the_end:
					x = tf.keras.layers.BatchNormalization(name="batch_norm_mine")(x)

				x = tf.keras.layers.Dense(self.reverse_y_map_length, activation=None, name="dense_mine")(x)
				model = tf.keras.models.Model(model.layers[0].input, x)

				if freeze:
					ii = 1
					if self.bn_at_the_end:
						ii += 1

					ii += 1  # for relu outputs

					for layer in model.layers:
						layer.trainable = False

					for nn in range(ii):
						model.layers[-(nn+1)].trainable = True

			model.summary()
			print(f"{self.name} loaded from {path}, please make sure that is what you want.")

		else:
			base_model = inception_resnet_v1.InceptionResNetV1(self.input_shape)

			x = self.pooling()(base_model.layers[-1].output)
			if dropout_rate > 0.0:
				x = tf.keras.layers.Dropout(dropout_rate)(x)

			x1 = tf.keras.layers.Dense(self.n_features, activation=None)(x)
			if self.mode == "softmax" or self.mode == "sparse softmax":
				x = tf.keras.layers.ReLU()(x1)
			if self.mode == "triplet":
				x = x1

			if self.bn_at_the_end:
				x = tf.keras.layers.BatchNormalization()(x)  # momentum=0.995, epsilon=0.001, scale=False,

			if self.mode == "softmax" or self.mode == "sparse softmax":
				x = tf.keras.layers.Dense(self.reverse_y_map_length, activation=None)(x)

			model = tf.keras.models.Model(base_model.layers[0].input, [x1, x])

			model.summary()
			print(f"{self.name} is created, didn't load from {path}, please make sure that is what you want.")

		model = self.make_model_ready(model, new_activation)
		self.model = model
		return model


if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

	md = MainData("../datasets", mnist_path="../datasets/mnist")
	md.run(real_examples = True, generated_examples = False, test_examples = False, mnist_examples=False, real_examples_will_be_reading=[
	"CASIA_NEW_MAXPY/", "105_classes_pins_dataset/"])

	# data_x, data_y = np.concatenate([md.g_real_paths, md.generated_paths]), np.concatenate([np.zeros((len(md.g_real_labels)), np.int32), np.ones((len(md.generated_paths)), np.int32)])
	# real_dataset_train, real_dataset_test = md.create_tensorflow_dataset_object(data_x, data_y, supportive=False)

	os.makedirs("processed_data", exist_ok=True)
	real_new_x, real_new_y = md.create_main_triplet_dataset(md.real_paths, md.real_labels, 200, data_path="processed_data/casia_and_mine_triplet.npy")

	triplet_dataset_train, triplet_dataset_test = md.create_tensorflow_dataset_object(real_new_x, real_new_y, supportive=False)
	# softmax_dataset_train, softmax_dataset_test = md.create_tensorflow_dataset_object(md.real_paths, md.real_labels, supportive=False)
	# mnist_dataset_train, mnist_dataset_test = md.create_tensorflow_dataset_object(md.mnist_paths, md.mnist_labels, supportive=False)

	xception_model = InceptionRV1(md, None, None, batch_size=16, epochs=10, mode="triplet", use_center_loss=False, selected_loss=None, y_map=md.real_y_map,
	 lr=0.0001, n_features=512, bn_at_the_end=False, input_shape=(128, 128, 3), pooling=tf.keras.layers.GlobalAveragePooling2D, new_name=None,
	  kernel_regularizer=tf.keras.regularizers.l2(5e-4))
	xception_model.get_model(dropout_rate=0.2, from_softmax=True, from_triplet=False, freeze=False)
	xception_model.train_loop(n=1000, use_accuracy=False)
