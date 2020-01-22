import tensorflow as tf


def xception(input_shape=(160, 160, 3)):
	model = tf.keras.applications.Xception(input_shape=input_shape, weights=None, include_top=False)

	return model