package TesterClasses;

import DataDisplay.DataReader;
import DataDisplay.DataSet;
import DataDisplay.Display;
import processing.core.PApplet;
import Perceptron.Perceptron;

import java.util.Collections;

public class DecisionBoundaryVizualizer extends PApplet {
    private static final int NO_CATEGORY_COLOR = 0xFFFFFF00;
    private static final int YES_CATEGORY_COLOR = 0xFFFF00FF;
    private static final int CORRECT_CLASSIFICAITON_COLOR = 0xFF00FF00;
    private static final int INCORRECT_CLASSIFICATION_COLOR = 0xFFFF0000;

    DataSet data;
    Perceptron perceptron;
    Display display;

    // Possible features: "sepal length", "sepal width", "petal length", "petal width"
    static String[] features = {"petal width", "sepal length"};     // <-- you can change this
    int x, y;

    int currentIndex = 0;

    public void settings() {
        size(800, 800);
    }

    public void setup() {
        perceptron = new Perceptron(2, "setosa");   // --=== [ you can change this ] ===--

        String[] headers = {"sepal length", "sepal width", "petal length", "petal width", "class"};
        data = DataReader.createDataSetFromCSV("data/iris.data", 0, headers);

        x = DataSet.getIndexForFeatureName(features[0]);
        y = DataSet.getIndexForFeatureName(features[1]);

        data.info();

        display = new Display(0, 0,
                width, height, data.getMinVal(x),
                data.getMinVal(y), data.getMaxVal(x),
                data.getMaxVal(y), 2f);
    }

    private static void testPerceptronOnData(Perceptron nn, DataSet d) {
        int numRight = 0;
        for (DataSet.DataPoint p : d.getData()) {
            String correctLabel = p.getLabelString();
            float[] input = p.getData(features);

            int guess = nn.guess(input);

            if (nn.isGuessCorrect(guess, correctLabel)) {
                numRight++;
            }

            String displayString = (guess == 1) ? "setosa" : "NOT setosa";
            System.out.println("Guessed: " + guess + " real: " + correctLabel);
        }

        System.out.println("Right: " + numRight + " / " + d.getData().size());
    }

    private static void runTrainingData(Perceptron nn, DataSet d) {
        for (int epochs = 0; epochs < 10; epochs++) {
            for (DataSet.DataPoint p : d.getData()) {
                String correctLabel = p.getLabelString();

                float[] input = p.getData(features);
                nn.train(input, correctLabel);
            }
        }
    }

    public void draw() {
        background(200);
        drawFullField(20);
        drawPoints();
        displayNNInfo(perceptron, 30, 30);
        // train(10);   <-- uncomment to train on 10 items every frame
    }

    private void displayNNInfo(Perceptron nn, int x, int y) {
        float[] weights = nn.getWeights();
        float threshold = nn.getThreshold();

        if (weights == null) return;

        String w1 = String.format("%.2f", weights[0]);
        String w2 = String.format("%.2f", weights[1]);
        String thresh = String.format("%.2f", threshold);
        String display = features[0] + "*" + w1 + " + "
                + features[1] + "*" + w2 + " >= " + thresh;

        strokeWeight(1);
        fill(255);
        rect(0, y - 24, width, 35);

        textSize(20);
        fill(0);
        stroke(0);
        text(display, x, y);
    }

    public void drawPoints() {
        for (DataSet.DataPoint point : data.getData()) {
            String label = point.getLabelString();

            int weight = 0;
            weight = 6;

            float[] inputs = point.getData(features);
            int guess = perceptron.guess(inputs);

            int color = (perceptron.isGuessCorrect(guess, label)) ? CORRECT_CLASSIFICAITON_COLOR : INCORRECT_CLASSIFICATION_COLOR;
            int stroke = (label.equals(perceptron.getTargetLabel())) ? YES_CATEGORY_COLOR : NO_CATEGORY_COLOR;
            display.plotDataCoords(this, point.getData(x), point.getData(y), 16, stroke, color, weight);
        }
    }

    private void drawFullField(int STEP) {
        for (int x = 0; x < width; x += STEP) {
            for (int y = 0; y < height; y += STEP) {
                float dx = display.screenXToData(x);
                float dy = display.sccreenYToData(y);

                int guess = perceptron.guess(new float[]{dx, dy});
                int color = (guess == 1) ? YES_CATEGORY_COLOR : NO_CATEGORY_COLOR;

                display.plotDataCoords(this, dx, dy, STEP / 2, color, color, 1);
            }
        }
    }

    public void train(int numItems) {
        Collections.shuffle(data.getData());
        for (int i = 0; i < numItems; i++) {
            DataSet.DataPoint point = data.getData().get(i);
            String label = point.getLabelString();

            float[] inputs = {point.getData(x), point.getData(y)};
            perceptron.train(inputs, point.getLabelString());
        }
    }

    public void mouseReleased() {
        train(data.getData().size());
    }

   public static void main(String[] args) {
        PApplet.main("TesterClasses.DecisionBoundaryVizualizer");
    }
}