
import java.util.Arrays;

/*
some utility functions for manipulating matrices
functions assume that matrices are represented "vertically": [n][m] represents the standard matrix with m rows and n columns
i.e. the transpose of the standard matrix
*/

public class Matrices {

	//computes AB
	public static double[][] multiply(double[][] A, double[][] B) {
		int m = A[0].length;
		int n = B.length;
		if (A.length != B[0].length) {
			throw new IllegalArgumentException();
		}
		double product[][] = new double[n][m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				double sum = 0;
				for (int k = 0; k < A.length; k++) {
					sum += A[k][i] * B[j][k];
				}
				product[j][i] = sum;
			}
		}
		return product;
	}

	public static double[][] transpose(double[][] matrix) {
		double[][] T = new double[matrix[0].length][matrix.length];
		for (int i = 0; i < matrix.length; i++)
			for (int j = 0; j < matrix[0].length; j++)
				T[j][i] = matrix[i][j];
		return T;
	}

	//computes A - B
	public static double[][] subtract(double[][] A, double[][] B) {
		double[][] difference = new double[A.length][A[0].length];
		for (int i = 0; i < A.length; i++)
			for (int j = 0; j < A[0].length; j++)
				difference[i][j] = A[i][j] - B[i][j];
		return difference;
	}

	public static double[][] scalarMultiply(double[][] matrix, double s) {
		double[][] result = new double[matrix.length][matrix[0].length];
		for (int i = 0; i < matrix.length; i++)
			for (int j = 0; j < matrix[0].length; j++)
				result[i][j] = s * matrix[i][j];
		return result;
	}

	//prints as standard form
	public static void print(double[][] matrix) {
		for (int i = 0; i < matrix[0].length; i++) {
			for (int j = 0; j < matrix.length; j++) {
				System.out.print(matrix[j][i] + " ");
			}
			System.out.println();
		}
	}

	public static void fill(double[][] matrix, double d) {
		for (int i = 0; i < matrix.length; i++)
			for (int j = 0; j < matrix[0].length; j++)
				matrix[i][j] = d;
	}

	public static void fillRand(double[][] matrix, double lower, double upper) {
		if (upper < lower)
			throw new IllegalArgumentException();
		for (int i = 0; i < matrix.length; i++)
			for (int j = 0; j < matrix[0].length; j++)
				matrix[i][j] = lower + Math.random() * (upper - lower);
	}

}