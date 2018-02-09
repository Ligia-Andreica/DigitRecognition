package neural;

/** A "neuron" that does no computation, for specifying inputs. Its output can be manually set. */
public class InputNeuron extends Neuron {

	protected InputNeuron() {
		super(new Neuron[0], null, 0.0);
	}
	
	/** Sets the output of this neuron. */
	public void setOutput(double output) {
		super.setOutput(output);
	}

	@Override
	public double squash(double sum) {
		// Irrelevant
		return -1;
	}

	@Override
	public void update() {
		// Does nothing
	}

}
