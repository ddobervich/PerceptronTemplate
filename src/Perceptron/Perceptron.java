package Perceptron;

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
        // TODO:  initialize the weights
        return null;
    }

    /***
     * Run the perceptron on the input and return 0 or 1 for the output category
     * @param input input vector
     * @return 0 or 1 representing the possible output categories or -1 if there's an error
     */
    public int guess(float[] input) {
        // TODO:  Implement this.
        // Do a linear combination of the inputs multiplied by the weights.
        // Run the sum through the activiationFunction and return the result
        return -1;
    }

    private int activationFunction(float sum) {
        if (sum > THRESHOLD) {
            return 1;
        } else {
            return 0;
        }
    }

    /***
     * Train the perceptron using the input feature vector and its correct label.
     * Return true if there was a non-zero error and training occured (weights got adjusted)
     *
     * @param input
     * @param correctLabel
     * @return
     */
    public boolean train(float[] input, String correctLabel) {
        // TODO:  Implement this.

        // run the perceptron on the input
        // compare the guess with the correct label (can use already-made helper method for this).

        // If guess was incorrect
        //    update weights and THRESHOLD using learning rule

        return false;
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