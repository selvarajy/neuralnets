import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

input_vec_size = 784
num_classes = output_vec_size = 10
num_epochs = 20
batch_size = 100
num_hidden_layers=3
hidden_layer_neurons = {1:500,2:500,3:500}

# Define the place holder to feed the input
# treat this as input layer
X = tf.placeholder('float32',[None,input_vec_size])

# Define output placeholder
Y = tf.placeholder('float32',[None,output_vec_size])

def build_network(data):
    network = {}
    for layer in range(1,num_hidden_layers+1):
        # Define weights and biases for each layer
        # random initialization
        network[layer] = {'weights':tf.Variable(tf.random_normal([input_vec_size if layer==1 else hidden_layer_neurons[layer-1],hidden_layer_neurons[layer]])),
                          'biases':tf.Variable(tf.random_normal([hidden_layer_neurons[layer]]))}
    # Final output layer
    network['output_layer'] = {'weights':tf.Variable(tf.random_normal([hidden_layer_neurons[layer],output_vec_size])),
                         'biases':tf.Variable(tf.random_normal([output_vec_size]))}
    
    # compute activations
    activations = None
    for layer in range(1, num_hidden_layers+1):
        activations = tf.nn.relu(tf.add(tf.matmul(data if layer==1 else activations, network[layer]['weights']),network[layer]['biases']))
    # compute final output
    return tf.add(tf.matmul(activations,network['output_layer']['weights']),network['output_layer']['biases'])

def train_nw(X):
    prediction = build_network(data=X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=Y))
    optimizer = tf.train.AdadeltaOptimizer().minimize(cost)
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train._num_examples/batch_size)):
                e_X,e_Y = mnist.train.next_batch(batch_size)
                _,c = session.run([optimizer,cost], feed_dict={X:e_X,Y:e_Y})
                epoch_loss+=c
            print('Epoch', epoch, 'completed out of',num_epochs,'loss:',epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))
        print('Accuracy:',accuracy.eval({X:mnist.test.images, Y:mnist.test.labels}))
                 
train_nw(X)
