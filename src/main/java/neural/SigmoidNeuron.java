package neural;

import java.util.Random;

/** A neuron with a logistic sigmoid activation function. */
public class SigmoidNeuron extends Neuron {

	/** Scaling constant for learning. */
	private static final double LEARNING_RATE = 0.1;

	/** Error of this unit. */
	private double delta;

	protected SigmoidNeuron(Neuron[] inputs) {
		// All weights (including the bias) should be distributed
		// uniformly in the range [-0.1, 0.1).
		super(inputs,null, 0.0);
		
		double rangeMin = -0.1;
		double rangeMax = 0.1;
		Random random = new Random();
		double[] weights = new double[inputs.length];
		for (int i = 0; i < inputs.length; i++) {
			weights[i] = rangeMin + (rangeMax - rangeMin) * random.nextDouble();
		}
		double bias = rangeMin + (rangeMax - rangeMin) * random.nextDouble();
		setWeights(weights, bias);
	}

	/** Returns the delta (error) value of this neuron. */
	protected double getDelta() {
		return delta;
	}

	/** Sets the delta (error) value of this neuron. */
	protected void setDelta(double delta) {
		this.delta = delta;
	}

	/** The sigmoid function */
	@Override
	public double squash(double sum) {
		return 1 / (1 + Math.pow(Math.E, -sum));
	}

	/**
	 * Updates the weights for this neuron to minimize error. Assumes delta has
	 * been set.
	 */
	public void updateWeights() {
		double[] weights = getWeights();
		for (int i = 0; i < weights.length; i++) {
			weights[i] += LEARNING_RATE * getInput(i).getOutput() * delta;
			increaseBias(-LEARNING_RATE * delta);
		}
	}
}
