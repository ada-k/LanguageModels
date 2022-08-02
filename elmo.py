# import necessary libraries
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Load pretrained ELMo model
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

# create an instance of ELMo
embeddings = elmo(
	[
		"I love to watch TV",
		"I am wearing a wrist watch"
	],
	signature="default",
	as_dict=True)["elmo"]
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Print word embeddings for word WATCH in given two sentences
print('Word embeddings for word WATCH in first sentence')
print(sess.run(embeddings[0][3]))
print('Word embeddings for word WATCH in second sentence')
print(sess.run(embeddings[1][5]))
