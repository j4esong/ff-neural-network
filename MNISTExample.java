
import java.io.*;
import java.util.Arrays;

class MNISTExample {

	static InputStream inputStream;

	public static void main(String[] args) throws FileNotFoundException, IOException {

		double[][] trainImages = loadImages(new File("mnist/train-images.idx3-ubyte"));
		double[][] trainLabels = oneHotDigit(loadLabels(new File("mnist/train-labels.idx1-ubyte")));
		double[][] testImages = loadImages(new File("mnist/t10k-images.idx3-ubyte"));
		double[] testLabels = toDoubleArray(loadLabels(new File("mnist/t10k-labels.idx1-ubyte")));

		FFNeuralNetwork net = new FFNeuralNetwork(784);

		net.addLayer(200, FFNeuralNetwork.ActivationFunction.sigmoid);
		net.addLayer(80, FFNeuralNetwork.ActivationFunction.sigmoid);
		net.addLayer(10, FFNeuralNetwork.ActivationFunction.sigmoid);

		net.fillRandWeights(-1, 1);
		net.train(trainImages, trainLabels, 1, 0.003, true);
		net.testClassifier(testImages, testLabels, false);
	}

	private static double[][] oneHotDigit(int[] x) {
		double[][] result = new double[x.length][10];
		for (int i = 0; i < x.length; i++) {
			result[i][x[i]] = 1;
		}
		return result;
	}

	private static double[][] loadImages(File file) throws FileNotFoundException, IOException {
		inputStream = new BufferedInputStream(new FileInputStream(file));
		inputStream.skip(4);
		int imageNum = nextNByte(4);
		int rows = nextNByte(4);
		int cols = nextNByte(4);
		double[][] images = new double[imageNum][rows * cols];
		for (int i = 0; i < imageNum; i++) {
			for (int k = 0; k < cols * rows; k++) {
				images[i][k] = nextNByte(1);
			}
		}
		inputStream.close();
		return images;
	}

	private static int[] loadLabels(File file) throws FileNotFoundException, IOException {
		inputStream = new BufferedInputStream(new FileInputStream(file));
		inputStream.skip(4);
		int labelNum = nextNByte(4);
		int[] labels = new int[labelNum];
		for (int i = 0; i < labelNum; i++)
			labels[i] = nextNByte(1);
		inputStream.close();
		return labels;
	}

	private static int nextNByte(int n) throws IOException {
		int k = inputStream.read() << ((n - 1) * 8);
		for (int i = n - 2; i >= 0; i--)
			k += inputStream.read() << (i * 8);
		return k;
	}

	private static double[] toDoubleArray(int[] array) {
		double[] result = new double[array.length];
		for (int i = 0; i < array.length; i++)
			result[i] = (double) array[i];
		return result;
	}

	//for testing basic functionality of neural network (learn 1 case)
	private static double[][] duplicateFirstLabel(double[][] labels) {
		double[][] testMatrix = new double[labels.length][labels[0].length];
		for (int i = 0; i < testMatrix.length; i++)
			testMatrix[i] = labels[0];
		return testMatrix;
	}

}