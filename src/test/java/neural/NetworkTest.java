package neural;

import junit.framework.TestCase;
import neural.Network;

import org.junit.Before;
import org.junit.Test;

public class NetworkTest extends TestCase {

	private Network net;

	@Before
	public void setUp() {
		net = new Network(2, 2, 1);
	}
	
	@Test
	public void testManualXor() {
		net.getNeuron(1, 0).setWeights(new double[] { 10.0, 10.0 }, 5.0);
		net.getNeuron(1, 1).setWeights(new double[] { -10.0, -10.0 }, -15.0);
		net.getNeuron(2, 0).setWeights(new double[] { 10.0, 10.0 }, 15.0);
		System.out.println(net.toString());
		for (int i = 0; i <= 1; i++) {
			for (int j = 0; j <= 1; j++) {
				assertEquals(i == j ? 0.0 : 1.0,
						net.run(new double[] { i, j })[0], 0.1);
			}
		}
	}

	@Test
	public void testLearnedXor() {
		// Train
		for (int epoch = 0; epoch < 100000; epoch++) {
			for (int i = 0; i <= 1; i++) {
				for (int j = 0; j <= 1; j++) {
					net.train(new double[] { i, j },
							new double[] { i == j ? 0.0 : 1.0 });
				}
			}
		}
		// Test
		for (int i = 0; i <= 1; i++) {
			for (int j = 0; j <= 1; j++) {
				assertEquals(i == j ? 0.0 : 1.0,
						net.run(new double[] { i, j })[0], 0.1);
			}
		}
	}

}
