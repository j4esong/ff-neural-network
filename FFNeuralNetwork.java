
//todo: implement other activations, figure out encapsulation stuff

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
			firstLayer = new Layer(new double[inputNodes + 1][neurons], null, null);
		} else {
			Layer currentLayer = getLayer(size);
			currentLayer.next = new Layer(new double[currentLayer.weights[0].length + 1][neurons], currentLayer, null);
		}
		size++;
	}

	//can users modify weights from reference?
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

	public void train(double[][] x, double[][] y, int passes, double step) {
		train(x, y, passes, step, false);
	}

	public void train(double[][] x, double[][] y, int passes, double step, boolean verbose) {
		this.step = step;
		for (int pass = 0; pass < passes; pass++) {
			for (int i = 0; i < x.length; i++) {
				double[][] phi = {x[i]};
				firstLayer.forward(Layer.append(phi, 1));
				Layer lastLayer = getLayer(size);
				if (verbose) {
					double loss = 0;
					double[][] residuals = Matrices.subtract(lastLayer.values, new double[][] {y[i]});	//im pretty sure this works
					for (int j = 0; j < residuals[0].length; j++)
						loss += Math.pow(residuals[0][j], 2);
					System.out.println((pass + 1) + "-" + i + ": " + loss);
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

		public double[][] weights;
		public double[][] values;

		public double[][] gradWeights;
		public double[][] input;		//to calculate gradient in backprop

		public Layer next;
		public Layer prev;

		public Layer(double[][] weights, Layer prev, Layer next) {
			this.weights = weights;
			this.prev = prev;
			this.next = next;
		}

		public void forward(double[][] input) {
			this.input = input;
			values = Matrices.multiply(weights, input);
			values = sigmoid(values);
			if (next != null)
				next.forward(append(values, 1));
		}

		public void backward(double[][] grad) {
			double[][] temp = dsigmoid(values);
			for (int i = 0; i < grad[0].length; i++) {
				grad[0][i] = grad[0][i] * temp[0][i];
			}
			gradWeights = Matrices.multiply(grad, Matrices.transpose(input));
			if (prev != null)
				prev.backward(Matrices.multiply(Matrices.transpose(weights), (next == null) ? grad : truncateLast(grad)));
		}

		public void update() {
			weights = Matrices.subtract(weights, Matrices.scalarMultiply(gradWeights, step));
		}

		//x must be a vector
		private double[][] sigmoid(double[][] x) {
			for (int i = 0; i < x[0].length; i++)
				x[0][i] = 1.0 / (1.0 + Math.exp(-1.0 * x[0][i]));
			return x;
		}

		//dsigmoid(x)/dx = (1 - sigmoid(x)) * sigmoid(x)
		private double[][] dsigmoid(double[][] sigmoid) {
			for (int i = 0; i < sigmoid[0].length; i++)
				sigmoid[0][i] = (1 - sigmoid[0][i]) * sigmoid[0][i];
			return sigmoid;
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