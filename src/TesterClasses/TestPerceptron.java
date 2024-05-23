package TesterClasses;

import DataDisplay.DataReader;
import DataDisplay.DataSet;
import Perceptron.Perceptron;

import java.util.ArrayList;
import java.util.List;

public class TestPerceptron {
    public static final String WHAT_TO_CLASSIFY = "versicolor";
    public static final String TRAINING_DATA_FILE = "data/iris.data";
    public static final String[] features =  {"petal length", "petal width", "sepal length", "sepal width"};

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
        for (int epochs = 0; epochs < 5000; epochs++) {
            trainN(nn, d, 100);
        }
    }

    private static void trainN(Perceptron nn, DataSet d, int num) {
        ArrayList<DataSet.DataPoint> batch = getRandomData(d.getData(), num);
        nn.train(batch, features);
    }

    private static ArrayList<DataSet.DataPoint> getRandomData(List<DataSet.DataPoint> data, int num) {
        ArrayList<DataSet.DataPoint> batch = new ArrayList<DataSet.DataPoint>();
        for (int i = 0; i < num; i++) {
            DataSet.DataPoint p = data.get((int)(Math.random()*data.size()));
            batch.add(p);
        }
        return batch;
    }

    private static void test(Perceptron nn, DataSet d) {
        int numRight = 0;
        for (DataSet.DataPoint p : d.getData()) {
            String correctLabel = p.getLabelString();

            float[] input = p.getData(features);
            float prob = nn.guess(input);
            int guess = 0;
            if (prob > 0.5) guess = 1;

            if (nn.isGuessCorrect(guess, correctLabel)) {
                numRight++;
            }

            String displayString = (guess == 1) ? WHAT_TO_CLASSIFY : "NOT " + WHAT_TO_CLASSIFY;
            System.out.println("Guessed: " + displayString + "\t\t real: " + correctLabel);
        }

        System.out.println("Right: " + numRight + " / " + d.getData().size());
    }


}