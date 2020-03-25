import numpy as np


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, 
	learning_rate, hidden_layers):
        """ Initializes the parameters and link weights of the network """
        
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        self.hlayers = hidden_layers


        # Initializes input-->hidden and hidden-->output weights.
        self.wih1 = np.random.normal(0.0, pow(self.inodes, -0.5), 
			(self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), 
			(self.onodes, self.hnodes))

        # Initializes weights for the number of hidden layers chosen.
        if self.hlayers > 1:
            self.wh1h2 = np.random.normal(0.0, pow(self.hnodes, -0.5), 
                (self.hnodes, self.hnodes))
            if self.hlayers > 2:
                self.wh2h3 = np.random.normal(0.0, pow(self.hnodes, -0.5), 
                    (self.hnodes, self.hnodes))
                if self.hlayers > 3:
                    self.wh3h4 = np.random.normal(0.0, pow(self.hnodes, 
                        -0.5), (self.hnodes, self.hnodes))
                    if self.hlayers > 4:
                        self.wh4h5 = np.random.normal(0.0, pow(self.hnodes, 
                            -0.5), (self.hnodes, self.hnodes))
                        if self.hlayers > 5:
                            print('Only hidden layers <= 5 are supported.')
        
        # Anon function to call the sigmoid activation function.
        # Would need to change the sigmoid deriv in backprop to change this.
        self.activation = lambda x : (1 / (1 + np.exp(-x)))


    def train(self, inputs_list, targets_list):
        """ Takes the inputs and target values for a network and forward
        propagates the data. It then compares target value with actual value
        and propagates the errors back through the nodes, updating weights.
        """

        # Creates 2-dimensional arrays from lists. Will prepend 1s to the 
		# array if not at least ndim=2.
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

		# Calcs dot product of input-hidden weights and input values, outputs
		# an array with summed input values for each hnode.
        hidden1_inputs = np.dot(self.wih1, inputs)
		# Applies activation function to summed inputs to a hnode. Can be 
		# eventually combined with above line.
        hidden1_outputs = self.activation(hidden1_inputs)
        
        # Performs matrix multiplication on a layer's weights and the last
        # layers outputs, then applies the activation function to the result.
        if self.hlayers > 1:
            hidden2_inputs = np.dot(self.wh1h2, hidden1_outputs)
            hidden2_outputs = self.activation(hidden2_inputs)
            if self.hlayers > 2:
                hidden3_inputs = np.dot(self.wh2h3, hidden2_outputs)
                hidden3_outputs = self.activation(hidden3_inputs)
                if self.hlayers > 3:
                    hidden4_inputs = np.dot(self.wh3h4, hidden3_outputs)
                    hidden4_outputs = self.activation(hidden4_inputs)
                    if self.hlayers > 4:
                        hidden5_inputs = np.dot(self.wh4h5, hidden4_outputs)
                        hidden5_outputs = self.activation(hidden5_inputs)


        if self.hlayers == 1:
            
		    # Calcs dot product of hidden-output weights and input values, 
            # outputs an array w/ summed output values for each onode. Applies
            # activation function.
            final_inputs = np.dot(self.who, hidden1_outputs)
            final_outputs = self.activation(final_inputs)
            # Calcs error between real answer and NN prediction.
            output_errors = targets - final_outputs
            # Proportionally calcs how much correct each hnode receives.
            hidden1_errors = np.dot(self.who.T, output_errors)
            # Updates hidden-output and input-hidden weights based on error.
            self.who += self.lr * np.dot((output_errors * final_outputs
                * (1.0 - final_outputs)), np.transpose(hidden1_outputs))       
            self.wih1 += self.lr * np.dot((hidden1_errors * hidden1_outputs
                * (1.0 - hidden1_outputs)), np.transpose(inputs))

        if self.hlayers == 2:

            final_inputs = np.dot(self.who, hidden2_outputs)
            final_outputs = self.activation(final_inputs)

            output_errors = targets - final_outputs

            hidden2_errors = np.dot(self.who.T, output_errors)
            hidden1_errors = np.dot(self.wh1h2.T, hidden2_errors)

            self.who += self.lr * np.dot((output_errors * final_outputs
                * (1.0 - final_outputs)), np.transpose(hidden2_outputs))
            self.wh1h2 += self.lr * np.dot((hidden2_errors * hidden2_outputs
                * (1.0 - hidden2_outputs)), np.transpose(hidden1_outputs))            
            self.wih1 += self.lr * np.dot((hidden1_errors * hidden1_outputs
                * (1.0 - hidden1_outputs)), np.transpose(inputs))

        if self.hlayers == 3:

            final_inputs = np.dot(self.who, hidden3_outputs)
            final_outputs = self.activation(final_inputs)

            output_errors = targets - final_outputs

            hidden3_errors = np.dot(self.who.T, output_errors)
            hidden2_errors = np.dot(self.wh2h3.T, hidden3_errors)
            hidden1_errors = np.dot(self.wh1h2.T, hidden2_errors)

            self.who += self.lr * np.dot((output_errors * final_outputs
                * (1.0 - final_outputs)), np.transpose(hidden3_outputs))
            self.wh2h3 += self.lr * np.dot((hidden3_errors * hidden3_outputs
                * (1.0 - hidden3_outputs)), np.transpose(hidden2_outputs)) 
            self.wh1h2 += self.lr * np.dot((hidden2_errors * hidden2_outputs
                * (1.0 - hidden2_outputs)), np.transpose(hidden1_outputs))            
            self.wih1 += self.lr * np.dot((hidden1_errors * hidden1_outputs
                * (1.0 - hidden1_outputs)), np.transpose(inputs))

        if self.hlayers == 4: 

            final_inputs = np.dot(self.who, hidden4_outputs)
            final_outputs = self.activation(final_inputs)

            output_errors = targets - final_outputs

            hidden4_errors = np.dot(self.who.T, output_errors)
            hidden3_errors = np.dot(self.wh3h4.T, hidden4_errors)
            hidden2_errors = np.dot(self.wh2h3.T, hidden3_errors)
            hidden1_errors = np.dot(self.wh1h2.T, hidden2_errors)

            self.who += self.lr * np.dot((output_errors * final_outputs
                * (1.0 - final_outputs)), np.transpose(hidden4_outputs))
            self.wh3h4 += self.lr * np.dot((hidden4_errors * hidden4_outputs
                * (1.0 - hidden4_outputs)), np.transpose(hidden3_outputs)) 
            self.wh2h3 += self.lr * np.dot((hidden3_errors * hidden3_outputs
                * (1.0 - hidden3_outputs)), np.transpose(hidden2_outputs)) 
            self.wh1h2 += self.lr * np.dot((hidden2_errors * hidden2_outputs
                * (1.0 - hidden2_outputs)), np.transpose(hidden1_outputs))            
            self.wih1 += self.lr * np.dot((hidden1_errors * hidden1_outputs
                * (1.0 - hidden1_outputs)), np.transpose(inputs))

        if self.hlayers == 5:     

            final_inputs = np.dot(self.who, hidden5_outputs)
            final_outputs = self.activation(final_inputs)

            output_errors = targets - final_outputs

            hidden5_errors = np.dot(self.who.T, output_errors)
            hidden4_errors = np.dot(self.wh4h5.T, hidden5_errors)
            hidden3_errors = np.dot(self.wh3h4.T, hidden4_errors)
            hidden2_errors = np.dot(self.wh2h3.T, hidden3_errors)
            hidden1_errors = np.dot(self.wh1h2.T, hidden2_errors)

            self.who += self.lr * np.dot((output_errors * final_outputs
                * (1.0 - final_outputs)), np.transpose(hidden5_outputs))
            self.wh4h5 += self.lr * np.dot((hidden5_errors * hidden5_outputs
                * (1.0 - hidden5_outputs)), np.transpose(hidden4_outputs)) 
            self.wh3h4 += self.lr * np.dot((hidden4_errors * hidden4_outputs
                * (1.0 - hidden4_outputs)), np.transpose(hidden3_outputs)) 
            self.wh2h3 += self.lr * np.dot((hidden3_errors * hidden3_outputs
                * (1.0 - hidden3_outputs)), np.transpose(hidden2_outputs)) 
            self.wh1h2 += self.lr * np.dot((hidden2_errors * hidden2_outputs
                * (1.0 - hidden2_outputs)), np.transpose(hidden1_outputs))            
            self.wih1 += self.lr * np.dot((hidden1_errors * hidden1_outputs
                * (1.0 - hidden1_outputs)), np.transpose(inputs))
          
        return output_errors


    def query(self, inputs_list):
        """ Forward propagates validation data through the network to check
        performance. Check train function above for comments on how commands
        below function.
        """

        inputs = np.array(inputs_list, ndmin=2).T

        if self.hlayers == 1:
            
            hidden1_inputs = np.dot(self.wih1, inputs)
            hidden1_outputs = self.activation(hidden1_inputs)
            final_inputs = np.dot(self.who, hidden1_outputs)
            final_outputs = self.activation(final_inputs)

        if self.hlayers == 2:
            
            hidden1_inputs = np.dot(self.wih1, inputs)
            hidden1_outputs = self.activation(hidden1_inputs)
            hidden2_inputs = np.dot(self.wh1h2, hidden1_outputs)
            hidden2_outputs = self.activation(hidden2_inputs)
            final_inputs = np.dot(self.who, hidden2_outputs)
            final_outputs = self.activation(final_inputs)

        if self.hlayers == 3:
            
            hidden1_inputs = np.dot(self.wih1, inputs)
            hidden1_outputs = self.activation(hidden1_inputs)
            hidden2_inputs = np.dot(self.wh1h2, hidden1_outputs)
            hidden2_outputs = self.activation(hidden2_inputs)
            hidden3_inputs = np.dot(self.wh2h3, hidden2_outputs)
            hidden3_outputs = self.activation(hidden3_inputs)
            final_inputs = np.dot(self.who, hidden3_outputs)
            final_outputs = self.activation(final_inputs)
        
        if self.hlayers == 4:
            
            hidden1_inputs = np.dot(self.wih1, inputs)
            hidden1_outputs = self.activation(hidden1_inputs)
            hidden2_inputs = np.dot(self.wh1h2, hidden1_outputs)
            hidden2_outputs = self.activation(hidden2_inputs)
            hidden3_inputs = np.dot(self.wh2h3, hidden2_outputs)
            hidden3_outputs = self.activation(hidden3_inputs)
            hidden4_inputs = np.dot(self.wh3h4, hidden3_outputs)
            hidden4_outputs = self.activation(hidden4_inputs)
            final_inputs = np.dot(self.who, hidden4_outputs)
            final_outputs = self.activation(final_inputs)

        if self.hlayers == 5:

            hidden1_inputs = np.dot(self.wih1, inputs)
            hidden1_outputs = self.activation(hidden1_inputs)
            hidden2_inputs = np.dot(self.wh1h2, hidden1_outputs)
            hidden2_outputs = self.activation(hidden2_inputs)
            hidden3_inputs = np.dot(self.wh2h3, hidden2_outputs)
            hidden3_outputs = self.activation(hidden3_inputs)
            hidden4_inputs = np.dot(self.wh3h4, hidden3_outputs)
            hidden4_outputs = self.activation(hidden4_inputs)
            hidden5_inputs = np.dot(self.wh4h5, hidden4_outputs)
            hidden5_outputs = self.activation(hidden5_inputs)
            final_inputs = np.dot(self.who, hidden5_outputs)
            final_outputs = self.activation(final_inputs)

        return final_outputs
