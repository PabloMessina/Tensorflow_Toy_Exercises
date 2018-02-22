import numpy as np
import tensorflow as tf
import sklearn as sk

def siamese_embedding(X, training):    
    with tf.variable_scope('embedding'):
        
        # input layer
        input_layer = tf.reshape(X, [-1,28,28,1])
        
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
        fc1 = tf.layers.dense(
            inputs=pool2_flat,
            units=1024,
            activation=tf.nn.relu,
            name='fc1')
        
        # dropout
        dropout = tf.layers.dropout(
            inputs=fc1,
            rate=0.4,
            training=training,
            name='dropout')

        # fully connected layer 2
        fc2 = tf.layers.dense(
            inputs=dropout,
            units=64,
            name='fc2')
        
        return fc2
    
def generate_left_right_label(X, Y, n_per_class, n_between_class):    
    
    n_tot  = 10 * n_per_class + n_between_class
    left = np.empty((n_tot, 784), dtype=float)
    right = np.empty((n_tot, 784), dtype=float)
    label = np.empty((n_tot,), dtype=float)
    
    digit2idxs = [ list() for d in range(10) ]
    for i, d in enumerate(Y):
        digit2idxs[d].append(i)
    
    cur_pos = 0
    used_pairs = set()
    
    # within-class pairs    
    for d in range(10):
        idxs = digit2idxs[d]
        n = len(idxs)
        for _ in range(n_per_class):
            while True:
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j:
                    continue
                lidx = idxs[i]
                ridx = idxs[j]
                if lidx > ridx:
                    lidx, ridx = ridx, lidx
                p = (lidx,ridx)
                if p in used_pairs:
                    continue
                used_pairs.add(p)
                left[cur_pos] = X[lidx]
                right[cur_pos] = X[ridx]
                label[cur_pos] = 1.
                cur_pos += 1
                break
    
    # between-class pairs
    for _ in range(n_between_class):
        while True:
            ld = np.random.randint(10)
            rd = np.random.randint(10)
            if ld == rd:
                continue
            l_idxs = digit2idxs[ld]
            r_idxs = digit2idxs[rd]
            lidx = l_idxs[np.random.randint(len(l_idxs))]
            ridx = r_idxs[np.random.randint(len(r_idxs))]
            if lidx > ridx:
                lidx, ridx = ridx, lidx
            p = (lidx,ridx)
            if p in used_pairs:
                continue
            used_pairs.add(p)
            left[cur_pos] = X[lidx]
            right[cur_pos] = X[ridx]
            label[cur_pos] = 0.
            cur_pos += 1
            break
    
    assert(len(used_pairs) == cur_pos)
    assert(cur_pos == n_tot)
    return left, right, label
    
    
def generate_siamese_train_test_data():
    
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    X_mnist_train = mnist.train.images
    Y_mnist_train = mnist.train.labels
    X_mnist_test = mnist.test.images
    Y_mnist_test = mnist.test.labels
    
    left_train, right_train, label_train =\
        generate_left_right_label(X_mnist_train, Y_mnist_train, 5000, 50000)
    
    left_test, right_test, label_test =\
        generate_left_right_label(X_mnist_test, Y_mnist_test, 500, 5000)
        
    return dict(
        left_train=left_train,
        right_train=right_train,
        label_train=label_train,
        left_test=left_test,
        right_test=right_test,
        label_test=label_test,
    )
    

def main(unused_argv):
    
     # --- GENERATE TRAIN AND TEST DATA -------
    
    data = generate_siamese_train_test_data()
    left_train = data['left_train']
    right_train = data['right_train']
    label_train = data['label_train']
    left_test = data['left_test']
    right_test = data['right_test']
    label_test = data['label_test']

    # --- BUILD NETWORK GRAPH -------
    
    # placeholders
    left_X = tf.placeholder(tf.float32, shape=[None, 784], name='left_X')
    right_X = tf.placeholder(tf.float32, shape=[None, 784], name='right_X')
    label = tf.placeholder(tf.float32, shape=[None], name='label')
    training = tf.placeholder(tf.bool, name='training')

    # siamese embeddings
    with tf.variable_scope('siamese') as scope:
        left_embed = siamese_embedding(left_X, training)
        scope.reuse_variables()
        right_embed = siamese_embedding(right_X, training)        

    # contrastive loss
    margin = 1.0       
    d_square = tf.reduce_sum(tf.square(left_embed - right_embed), 1)
    d = tf.sqrt(d_square)
    contr_loss = (label * d_square) + ((1 - label) * tf.square(tf.maximum(0., margin - d)))
    contr_loss = 0.5 * tf.reduce_mean(contr_loss)

    # optimizer
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(contr_loss)
    
    # ------- TRAIN MODEL ---------
    
    num_epochs = 30
    minibatch_size = 256
    m = left_train.shape[0]
    
    num_minibatches = m // minibatch_size + int(m % minibatch_size > 0)
        
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(num_epochs):
            
            epoch_loss = 0.
            
            # shuffle train data
            left_train_shuff, right_train_shuff, label_train_shuff =\
                sk.utils.shuffle(left_train, right_train, label_train)
            
            for i in range(num_minibatches):
                
                start = i * minibatch_size
                end = start + minibatch_size
                left_train_mini = left_train_shuff[start:end]
                right_train_mini = right_train_shuff[start:end]
                label_train_mini = label_train_shuff[start:end]
                
                _, minibatch_loss = sess.run([optimizer, contr_loss],
                                             feed_dict={left_X: left_train_mini,
                                                        right_X: right_train_mini,
                                                        label: label_train_mini,
                                                        training: True})
                
                epoch_loss += minibatch_loss
            
            epoch_loss /= num_minibatches
            test_loss = contr_loss.eval(feed_dict={left_X: left_test,
                                                 right_X: right_test,
                                                 label: label_test,
                                                 training: False})
            print('epoch = %d, epoch_loss = %g, test_loss = %g' % (epoch, epoch_loss, test_loss))
        
        # ------- SAVE MODEL ---------
        
        save_path = saver.save(sess, "./tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)
    

if __name__ == "__main__":
    tf.app.run()