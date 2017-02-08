# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)
### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1], X_train.shape[2]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#plt.imshow(X_train[0])
#plt.show()

### Preprocess the data here.
### Feel free to use as many code cells as needed.

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

### Define your architecture here.
### Feel free to use as many code cells as needed.

import tensorflow as tf
from tensorflow.contrib.layers import flatten

EPOCHS = 10
BATCH_SIZE = 128

# Convolution help function
def conv2d(x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Pooling help function
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

def fully_connected(x, w, b):
    x = tf.matmul(x, w) + b
    return tf.nn.relu(x)

def LelNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # Layer 1: Conv. Input = 32x32x3. Output = .
    conv1 = conv2d(x, tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)), 
                  tf.Variable(tf.zeros(6)), 1)
    
    # Pooling 1: Input = 28x28x6. Output = 14x14x6
    pool1 = maxpool2d(conv1, 2)
    
    # Layer 2: Conv. Input = 10x10x16. Output = 5x5x16.
    conv2 = conv2d(pool1, tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma)), 
                  tf.Variable(tf.zeros(16)), 1)
    
    # Pooling 2: Input = 10x10x16. Output = 5x5x16
    pool2 = maxpool2d(conv2, 2)
    
    # Flatten. Input = 5x5x16. Output = 400
    flat = flatten(pool2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    full1 = fully_connected(flat, tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev = sigma)),
                           tf.Variable(tf.zeros(120)))
    
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    full2 = fully_connected(full1, tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev = sigma)),
                           tf.Variable(tf.zeros(84)))
    
    # Layer 5: Fully Connected. Input = 84. Output = 43.
    outW = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev = sigma))
    outB = tf.Variable(tf.zeros(43))
    out = tf.matmul(full2, outW) + outB
    
    return out

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
num_labels = 43
sparse_labels = tf.reshape(y, [-1, 1])
derived_size = tf.shape(sparse_labels)[0]
indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
concated = tf.concat(1, [indices, sparse_labels])
outshape = tf.concat(0, [tf.reshape(derived_size, [1]), tf.reshape(num_labels, [1])])
one_hot_y = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)

rate = 0.001

logits = LelNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
        

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        
        validation_accuracy = evaluate(X_validation, y_validation)
        print("Epoch {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    saver.save(sess, 'C:\CarND-Traffic-Sign-Classifier-Project\lalnerd')
    print("Model Saved")