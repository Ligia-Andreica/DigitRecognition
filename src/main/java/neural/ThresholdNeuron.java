package neural;

/** A linear threshold neuron. */
public class ThresholdNeuron extends Neuron {

	protected ThresholdNeuron(Neuron[] inputs, double[] weights, double bias) {
		super(inputs, weights, bias);
	}

	@Override
	public double squash(double sum) {
		return sum > 0 ? 1 : 0;
	}

}
