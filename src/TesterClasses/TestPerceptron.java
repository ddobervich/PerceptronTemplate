package TesterClasses;

import DataDisplay.DataReader;
import DataDisplay.DataSet;
import Perceptron.Perceptron;

public class TestPerceptron {
    public static final String WHAT_TO_CLASSIFY = "setosa";
    public static final String TRAINING_DATA_FILE = "data/iris.data";
    public static final String[] features = {"sepal length", "sepal width"};

    public static void main(String[] args) {
        DataSet dataset;
        Perceptron nn;

        String[] headers = {"sepal length", "sepal width", "petal length", "petal width", "class"};
        dataset = DataReader.createDataSetFromCSV(TRAINING_DATA_FILE, 0, headers);

        int numInputs = features.length;
        nn = new Perceptron(numInputs, WHAT_TO_CLASSIFY);

        train(nn, dataset);
        test(nn, dataset);
    }

    private static void train(Perceptron nn, DataSet d) {
        for (int epochs = 0; epochs < 500; epochs++) {
            for (DataSet.DataPoint p : d.getData()) {
                String correctLabel = p.getLabelString();
                float[] input = p.getData(features);

                nn.train(input, correctLabel);
            }
        }
    }

    private static void test(Perceptron nn, DataSet d) {
        int numRight = 0;
        for (DataSet.DataPoint p : d.getData()) {
            String correctLabel = p.getLabelString();

            float[] input = p.getData(features);
            int guess = nn.guess(input);

            if (nn.isGuessCorrect(guess, correctLabel)) {
                numRight++;
            }

            String displayString = (guess == 1) ? WHAT_TO_CLASSIFY : "NOT " + WHAT_TO_CLASSIFY;
            System.out.println("Guessed: " + displayString + "\t\t real: " + correctLabel);
        }

        System.out.println("Right: " + numRight + " / " + d.getData().size());
    }


}