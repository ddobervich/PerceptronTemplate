package Perceptron;

import DataDisplay.DataSet;
import sun.security.krb5.internal.TGSReq;

import java.util.ArrayList;
import java.util.Arrays;

public class Perceptron {
    private int numInputs;
    private float[] weights;
    private float THRESHOLD = 0;

    private String classifyForLabel; // this is a classifier for eg "virginica"
    private float learningRate = 0.005f;

    public Perceptron(int numInputs, String whatToClassify) {
        this.classifyForLabel = whatToClassify;
        this.numInputs = numInputs;
        weights = initWeights(numInputs);
    }

    /***
     * Create and return a float array with randomly initialized weights from [-1 to 1]
     * @param numInputs the number of weights needed (length of the array to create)
     * @return the initialized weights array
     */
    private float[] initWeights(int numInputs) {
        float[] weights = new float[numInputs];

        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float)(Math.random()*2 - 1);
        }

        return weights;
    }

    /***
     * Run the perceptron on the input and return 0 or 1 for the output category
     * @param input input vector
     * @return 0 or 1 representing the possible output categories or -1 if there's an error
     */
    public float guess(float[] input) {
        float sum = 0;

        for (int i = 0; i < input.length; i++) {
            sum += input[i]*weights[i];
        }

        sum += THRESHOLD;

        return activationFunction(sum);
    }

    private float activationFunction(float sum) {
        return (float)(1.0/(1+Math.exp(-sum)));
    }

    /***
     * Train the perceptron using the input feature vector and its correct label.
     * Return true if there was a non-zero error and training occured (weights got adjusted)
     * @return
     */
    public boolean train(ArrayList<DataSet.DataPoint> batch, String[] features) {
        float[] weightUpdates = new float[ features.length ];
        float thresholdUpdate = 0;

        for (DataSet.DataPoint point : batch) {
            float[] input = point.getData(features);
            String correctLabel = point.getLabelString();

            float prediction = guess(input);
            int correctAnswer = getCorrectGuess(correctLabel);
            float error = (prediction - correctAnswer);

            for (int i = 0; i < weights.length; i++) {
                weightUpdates[i] += input[i] * error * learningRate;		// ADD IN ADJUSTMENTS FOR UDPATE
            }

            thresholdUpdate = thresholdUpdate + error * learningRate;
        }

        for (int i = 0; i < weights.length; i++) {					// APPLY THEM!
            weights[i] -= weightUpdates[i];
        }

        THRESHOLD -= thresholdUpdate;					// APPLY THEM!

        return true;
    }


    public float[] getWeights() {
        return weights;
    }

    public String getTargetLabel() {
        return this.classifyForLabel;
    }

    public boolean isGuessCorrect(int guess, String correctLabel) {
        return guess == getCorrectGuess(correctLabel);
    }

    /***
     * Return the correct output for a given class label.  ie returns 1 if input label matches
     * what this perceptron is trying to classify.
     * @param label
     * @return
     */
    public int getCorrectGuess(String label) {
        if (label.equals(this.classifyForLabel))
            return 1;
        else
            return 0;
    }

    public float getThreshold() {
        return THRESHOLD;
    }
}