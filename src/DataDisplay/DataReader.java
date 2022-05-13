package DataDisplay;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

public class DataReader {

	private static String normalizeLineBreaks(String s) {
		return s.replace("\r\n", "\n").replace('\r', '\n');
	}

	public static String readFileAsString(String filepath) {
		Scanner scanner = null;
		try {
			scanner = new Scanner( new File(filepath) );
			String text = scanner.useDelimiter("\\A").next();
			return text;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} finally {
			if (scanner != null)
				scanner.close(); // Put this call in a finally block
		}

		System.err.println("-------------------------------------------------------");
		System.err.println("There was an error reading your data file: " + filepath);
		System.err.println("Check the file path!");
		System.err.println("-------------------------------------------------------");

		return null;
	}

	public static DataSet createDataSetFromCSV(String filepath, int linesToSkip, String[] fileColumnHeaders) {
		String data = normalizeLineBreaks(readFileAsString(filepath));
		String[] lines = data.split("\n");

		int numColumns = fileColumnHeaders.length - 1; // last column is not a numeric column
		int startRow = linesToSkip;

		// create storage for data
		String[] dataLabels = Arrays.copyOf(fileColumnHeaders, fileColumnHeaders.length - 1);
		DataSet dataset = new DataSet(dataLabels);

		for (int r = startRow; r < lines.length; r++) {
			String line = lines[r];
			String[] coords = line.split(",");

			float[] datavals = new float[numColumns];
			for (int j = 0; j < numColumns; j++) {
				float val = 0;
				try {
					val = Float.parseFloat(coords[j]);
				} catch (Exception e) {
					System.err.println("Warning: invalid number format");
				}
				datavals[j] = val;
			}

			String label = coords[numColumns]; // column after numbers is label
			dataset.add(datavals, label);
		}

		return dataset;
	}
}