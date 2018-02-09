package neural;

import junit.framework.TestCase;

import neural.InputNeuron;
import neural.Neuron;
import neural.ThresholdNeuron;

import org.junit.Before;
import org.junit.Test;

public class ThresholdNeuronTest extends TestCase{

	private ThresholdNeuron neuron;
	
	@Before
	public void setUp() {
	}

	@Test
	public void testAnd() {
		InputNeuron a = new InputNeuron();
		InputNeuron b = new InputNeuron();
		neuron = new ThresholdNeuron(new Neuron[] {a, b}, new double[] {1.0, 1.0}, 1.5);
		for (int i = 0; i <= 1; i++) {
			for (int j = 0; j <= 1; j++) {
				a.setOutput(i);
				b.setOutput(j);
				neuron.update();
				assertEquals(i * j, neuron.getOutput(), 0.1);
			}
		}
	}
	
	@Test
	public void testNot() {
		InputNeuron a = new InputNeuron();
		neuron = new ThresholdNeuron(new Neuron[] {a}, new double[] {-2.0}, -1.0);
		a.setOutput(0.0);
		neuron.update();
		assertEquals(1.0, neuron.getOutput(), 0.1);
		a.setOutput(1.0);
		neuron.update();
		assertEquals(0.0, neuron.getOutput(), 0.1);
	}

	@Test
	public void testOr() {
		InputNeuron a = new InputNeuron();
		InputNeuron b = new InputNeuron();
		neuron = new ThresholdNeuron(new Neuron[] {a, b}, new double[] {1.0, 1.0}, 0.5);
		for (int i = 0; i <= 1; i++) {
			for (int j = 0; j <= 1; j++) {
				a.setOutput(i);
				b.setOutput(j);
				neuron.update();
				assertEquals((i + j > 0) ? 1.0 : 0.0, neuron.getOutput(), 0.1);
			}
		}
	}

}
