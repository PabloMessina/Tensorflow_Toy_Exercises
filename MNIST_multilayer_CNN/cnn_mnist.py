import numpy as np
import tensorflow as tf
import sklearn as sk

def main(unused_argv):
    
     # --- LOAD TRAIN AND TEST DATA -------
    
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    X_train = mnist.train.images  # Returns np.array
    Y_train = np.asarray(mnist.train.labels, dtype=np.int32)
    X_test = mnist.test.images  # Returns np.array
    Y_test = np.asarray(mnist.test.labels, dtype=np.int32)

    # --- BUILD NETWORK GRAPH -------

    # placeholders
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.int32, shape=[None])
    training = tf.placeholder(tf.bool, name='training')
    
    # input layer
    input_layer = tf.reshape(x, [-1,28,28,1])
    
    # convolution and maxpooling 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu,
        name='conv1')    
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=2,
        name='pool1')
    
    # convolution and maxpooling 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu,
        name='conv2')    
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2],
        strides=2,
        name='pool2')    
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64], name='pool2_flat')
    
    # fully connected layer 1
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu,
        name='fc1')    
    
    # dropout
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training=training,
        name='dropout')
    
    # logits layer (for prediction)
    logits = tf.layers.dense(
        inputs=dropout,
        units=10,
        name='logits')
    
    # class predictions - argmax
    predicted_classes = tf.argmax(input=logits, axis=1,
                                 name='predicted_classes',
                                 output_type=tf.int32)
    
    # class probabilities - softmax
    probabilities = tf.nn.softmax(logits,
                                  name='softmax_tensor')
    
    # cross entropy loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=y, logits=logits)
    
    # optimizer
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    # accuracy
    correct_prediction = tf.equal(predicted_classes, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # ------- TRAIN MODEL ---------
    
    num_epochs = 40
    minibatch_size = 128
    m = X_train.shape[0]
    
    num_minibatches = m // minibatch_size
    if m % minibatch_size > 0:
        num_minibatches += 1
        
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(num_epochs):
            
            epoch_loss = 0.
            
            # shuffle train data
            X_train_shuff, Y_train_shuff = sk.utils.shuffle(X_train, Y_train)            
            
            for i in range(num_minibatches):
                
                start = i * minibatch_size
                end = start + minibatch_size
                X_train_mini = X_train_shuff[start:end]
                Y_train_mini = Y_train_shuff[start:end]
                
                _, minibatch_loss = sess.run([optimizer, loss],
                                             feed_dict={x: X_train_mini, y: Y_train_mini, training: True})
                
                epoch_loss += minibatch_loss
            
            epoch_loss /= num_minibatches
            test_accuracy = accuracy.eval(feed_dict={x: X_test, y: Y_test, training: False})
            print('epoch = %d, loss = %g, test_accuracy = %g' % (epoch, epoch_loss, test_accuracy))
        
        save_path = saver.save(sess, "./tmp/minst_convnet_model.ckpt")
        print("Model saved in path: %s" % save_path)
    

if __name__ == "__main__":
    tf.app.run()