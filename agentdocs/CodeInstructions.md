Implement a feed-forward neural network
The model should have one hidden layer and be capable of being initialized with any number of input, hidden and output units/vertices. 
Use ReLU activation in the hidden layer and use a linear output layer (no activation function). 
The input and hidden layers should always both include a bias unit (how you implement the bias is your choice). 
Prior to training, set np.random.seed(0) and then initialize the model weights from a uniform random distribution U(-0.01, 0.01). 
For the training algorithm, use mini-batch SGD with batch size of 32, and re-shuffle training data
at the start of each epoch. Make sure you have set np.random.seed(0) to obtain consistent
results. Do not use regularization, momentum, or adaptive optimizers: use plain mini-batch SGD
only.
When implementing softmax, ensure numerical stability by subtracting the maximum logit before exponentiation. Links to the training datasets you are expected to use are elaborated further in the written section below.

Your implementation will be manually reviewed for completeness, as well as good programming practices. One crucial practice will be vectorized matrix operations, with others including choosing meaningful variable names, structuring reused code into functions or classes, and commenting your code to explain any tricky or important logic.
