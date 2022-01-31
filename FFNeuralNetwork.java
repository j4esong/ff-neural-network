
//todo: test all activations, implement other losses + mbsgd, gd

import java.util.Arrays;

class FFNeuralNetwork {

	public static double step = 0;
	public int inputNodes;
	public enum ActivationFunction {sigmoid, tanh, softmax, ReLU};

	private Layer firstLayer;
	private int size;

	public FFNeuralNetwork(int inputNodes) {
		this.inputNodes = inputNodes;
	}

	public void addLayer(int neurons, ActivationFunction f) {
		if (size == 0) {
			firstLayer = new Layer(new double[inputNodes + 1][neurons], null, null, f);
		} else {
			Layer currentLayer = getLayer(size);
			currentLayer.next = new Layer(new double[currentLayer.weights[0].length + 1][neurons], currentLayer, null, f);
		}
		size++;
	}

	public double[][] getLayerWeights(int layer) {
		return getLayer(layer).weights;
	}

	public void setLayerWeights(int layer, double[][] weights) {
		Layer currentLayer = getLayer(layer);
		if (!(currentLayer.weights.length == weights.length && currentLayer.weights[0].length == weights[0].length))
			throw new IllegalArgumentException("Matrix dimensions must match");
		currentLayer.weights = weights;
	}

	public void fillRandWeights(int layer, double lower, double upper) {
		Matrices.fillRand(getLayer(layer).weights, lower, upper);
	}

	public void fillRandWeights(double lower, double upper) {
		Layer currentLayer = firstLayer;
		for (int i = 0; i < size; i++) {
			Matrices.fillRand(currentLayer.weights, lower, upper);
			currentLayer = currentLayer.next;
		}
	}

	public void train(double[][] x, double[][] y, int epochs, double step) {
		train(x, y, epochs, step, false);
	}

	public void train(double[][] x, double[][] y, int epochs, double step, boolean verbose) {
		this.step = step;
		for (int epoch = 0; epoch < epochs; epoch++) {
			for (int i = 0; i < x.length; i++) {
				double[][] phi = {x[i]};
				firstLayer.forward(Layer.append(phi, 1));
				Layer lastLayer = getLayer(size);
				if (verbose) {
					double loss = 0;
					double[][] residuals = Matrices.subtract(lastLayer.values, new double[][] {y[i]});
					for (int j = 0; j < residuals[0].length; j++)
						loss += Math.pow(residuals[0][j], 2);
					System.out.println((epoch + 1) + "-" + i + ": " + loss);
				}
				lastLayer.backward(Matrices.scalarMultiply(Matrices.subtract(lastLayer.values, new double[][] {y[i]}), 2));
				updateAll();
			}
		}
	}

	public void testClassifier(double[][] x, double[] y) {
		testClassifier(x, y, false);
	}

	public void testClassifier(double[][] x, double[] y, boolean verbose) {
		int correct = 0;
		for (int i = 0; i < x.length; i++) {
			double[][] phi = {x[i]};
			firstLayer.forward(Layer.append(phi, 1));
			Layer lastLayer = getLayer(size);
			double max = -Double.MAX_VALUE;
			int prediction = 0;
			for (int j = 0; j < lastLayer.values[0].length; j++) {
				if (lastLayer.values[0][j] > max) {
					max = lastLayer.values[0][j];
					prediction = j;
				}
			}
			if (prediction == y[i])
				correct++;
			if (verbose)
				System.out.println(i + ": " + (prediction == y[i]));
		}
		System.out.println("accuracy: " + (double) correct / (double) x.length);
	}

	//1 based indexing
	private Layer getLayer(int layer) {
		if (layer > size || layer == 0)
			throw new IllegalArgumentException("No such layer");
		Layer currentLayer = firstLayer;
		for (int i = 0; i < layer - 1; i++)
			currentLayer = currentLayer.next;
		return currentLayer;
	}

	private void updateAll() {
		Layer currentLayer = firstLayer;
		for (int i = 0; i < size; i++) {
			currentLayer.update();
			currentLayer = currentLayer.next;
		}
	}

	private static class Layer {

		private double[][] weights;
		private double[][] values;

		private double[][] gradWeights;
		//input received from prev, including weight
		private double[][] input;

		private Layer next;
		private Layer prev;

		private ActivationFunction activation;

		private Layer(double[][] weights, Layer prev, Layer next, ActivationFunction activation) {
			this.weights = weights;
			this.prev = prev;
			this.next = next;
			this.activation = activation;
		}

		private void forward(double[][] input) {
			this.input = input;
			values = Matrices.multiply(weights, input);
			switch (activation) {
			case sigmoid:
				values = sigmoid(values);
				break;
			case ReLU:
				values = ReLU(values);
				break;
			case softmax:
				values = softmax(values);
				break;
			case tanh:
				values = dtanh(values);
				break;
			}
			if (next != null)
				next.forward(append(values, 1));
		}

		private void backward(double[][] upstreamGrad) {
			//what to initialize localGrad to? can't be dsigmoid because dsigmoid modifies values
			double[][] localGrad = values;
			switch (activation) {
			case sigmoid:
				localGrad = dsigmoid(values);
				break;
			case ReLU:
				localGrad = dReLU(values);
				break;
			case softmax:
				localGrad = dsoftmax(values);
				break;
			case tanh:
				localGrad = dtanh(values);
				break;
			}
			for (int i = 0; i < upstreamGrad[0].length; i++)
				upstreamGrad[0][i] = upstreamGrad[0][i] * localGrad[0][i];
			gradWeights = Matrices.multiply(upstreamGrad, Matrices.transpose(input));
			if (prev != null)
				prev.backward(Matrices.multiply(Matrices.transpose(weights), (next == null) ? upstreamGrad : truncateLast(upstreamGrad)));
		}

		private void update() {
			weights = Matrices.subtract(weights, Matrices.scalarMultiply(gradWeights, step));
		}

		/*
		since values field has already been updated with activation in forward pass,
		dReLU and dsigmoid, etc. have to be in terms of ReLU and sigmoid

		all activation functions modify the reference (values field) and only take vector inputs
		*/

		private static double[][] sigmoid(double[][] x) {
			for (int i = 0; i < x[0].length; i++)
				x[0][i] = 1.0 / (1.0 + Math.exp(-1.0 * x[0][i]));
			return x;
		}

		private static double[][] dsigmoid(double[][] sigmoid) {
			for (int i = 0; i < sigmoid[0].length; i++)
				sigmoid[0][i] = (1 - sigmoid[0][i]) * sigmoid[0][i];
			return sigmoid;
		}

		private static double[][] ReLU(double[][] x) {
			for (int i = 0; i < x[0].length; i++)
				x[0][i] = Math.max(0, x[0][i]);
			return x;
		}

		private static double[][] dReLU(double[][] x) {
			for (int i = 0; i < x[0].length; i++)
				if (x[0][i] != 0)
					x[0][i] = 1;
			return x;
		}

		public static double[][] softmax(double[][] x) {
			double denom = 0;
			for (int i = 0; i < x[0].length; i++) {
				x[0][i] = Math.exp(x[0][i]);
				denom += x[0][i];
			}
			for (int i = 0; i < x[0].length; i++)
				x[0][i] /= denom;
			return x;
		}

		public static double[][] dsoftmax(double[][] x) {
			return x;
		}

		public static double[][] tanh(double[][] x) {
			for (int i = 0; i < x[0].length; i++)
				x[0][i] = (Math.exp(2 * x[0][i]) - 1) / (Math.exp(2 * x[0][i]) + 1);
			return x;
		}

		public static double[][] dtanh(double[][] x) {
			for (int i = 0; i < x[0].length; i++)
				x[0][i] = 1 - Math.pow(x[0][i], 2);
			return x;
		}

		private static double[][] append(double[][] x, double e) {
			x[0] = Arrays.copyOf(x[0], x[0].length + 1);
			x[0][x[0].length - 1] = e;
			return x;
		}

		private static double[][] truncateLast(double[][] x) {
			x[0] = Arrays.copyOf(x[0], x[0].length - 1);
			return x;
		}
	}

}