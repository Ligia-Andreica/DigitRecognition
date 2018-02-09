package neural;

/** A feed-forward network with input, hidden, and output layers. */
public class Network {

	/** The input neurons. */
	private InputNeuron[] inputLayer;

	/** The hidden neurons. */
	private SigmoidNeuron[] hiddenLayer;

	/** The output neurons. */
	private SigmoidNeuron[] outputLayer;

	/**
	 * @param in
	 *            Number of input neurons.
	 * @param hid
	 *            Number of hidden neurons.
	 * @param out
	 *            Number of output neurons.
	 */
	protected Network(int in, int hid, int out) {
		inputLayer = new InputNeuron[in];
		for (int i = 0; i < in; i++)
			inputLayer[i] = new InputNeuron();
		
		hiddenLayer = new SigmoidNeuron[hid];
		for (int i = 0; i < hid; i++)
			hiddenLayer[i] = new SigmoidNeuron(inputLayer);
		
		outputLayer = new SigmoidNeuron[out];
		for (int i = 0; i < out; i++)
			outputLayer[i] = new SigmoidNeuron(hiddenLayer);
	}

	/**
	 * Returns the specified neuron. The input layer is layer 0, hidden 1,
	 * output 2.
	 */
	public Neuron getNeuron(int layer, int index) {
		switch (layer) {
			case 0: return inputLayer[index];
			case 1: return hiddenLayer[index];
			case 2: return outputLayer[index];
			default: return null;
		}
	}

	/**
	 * Returns the sum, over a set of training examples and across all outputs,
	 * of the square of the difference between actual and correct outputs. If
	 * learning is working properly, this should decrease over the course of
	 * training.
	 */
	public double meanSquaredError(double[][] inputs, double[][] correct) {
		double sum = 0.0;
		for (int i = 0; i < inputs.length; i++) {
			run(inputs[i]);
			for (int j = 0; j < outputLayer.length; j++) {
				sum += Math.pow(correct[i][j] - outputLayer[j].getOutput(), 2);
			}
		}
		return sum / (inputs.length * outputLayer.length);
	}

	/** Feeds inputs through the network, updating the output of each neuron. */
	public double[] run(double[] inputs) {
		for (int i = 0; i < inputs.length; i++)
			inputLayer[i].setOutput(inputs[i]);
		for (int i = 0; i < hiddenLayer.length; i++)
			hiddenLayer[i].update();
		for (int i = 0; i < outputLayer.length; i++)
			outputLayer[i].update();
		double[] result = new double[outputLayer.length];
		for (int i = 0; i < outputLayer.length; i++)
			result[i] = outputLayer[i].getOutput();
		return result;
	}

	@Override
	public String toString() {
		String result = "";
		result += "OUTPUT UNITS:\n";
		for (int i = 0; i < outputLayer.length; i++) {
			result += i + ": " + outputLayer[i] + "\n";
		}
		result += "HIDDEN UNITS:\n";
		for (int i = 0; i < hiddenLayer.length; i++) {
			result += i + ": " + hiddenLayer[i] + "\n";
		}
		return result + "(" + inputLayer.length + " INPUT UNITS)\n";
	}

	/**
	 * Slightly modifies this network's weights to cause it to response to
	 * inputs with something closer to the correct outputs.
	 */
	public void train(double[] inputs, double[] correct) {
		// This is a long method, with the following steps:

		// Feed the input forward through the network
		run(inputs);
		// Update deltas for output layer
		for (int i = 0; i < outputLayer.length; i++) {
			double error = outputLayer[i].getOutput()
					* (1 - outputLayer[i].getOutput())
					* (correct[i] - outputLayer[i].getOutput());
			outputLayer[i].setDelta(error);
		}
		// Update weights for output layer
		for (int i = 0; i < outputLayer.length; i++) {
			outputLayer[i].updateWeights();
		}
		// Update deltas for hidden layer
		for (int k = 0; k < hiddenLayer.length; k++) {
			double g = 0.0;
			for (int i = 0; i < outputLayer.length; i++) {
				double w = (outputLayer[i].getWeights())[k];
				g += outputLayer[i].getDelta() * w;
			}
			double error = hiddenLayer[k].getOutput()
					* (1 - hiddenLayer[k].getOutput()) * g;
			hiddenLayer[k].setDelta(error);
		}
		// Update weights for hidden layer
		for (int i = 0; i < hiddenLayer.length; i++) {
			hiddenLayer[i].updateWeights();
		}
	}
}
