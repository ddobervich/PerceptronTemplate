package Perceptron;

public class Perceptron {
    private int numInputs;
    private float[] weights;
    private float THRESHOLD = 0;

    private String labelToPredict;          // for example: this is a classifier for "virginica"
    private float learningRate = 0.005f;

    public Perceptron(int numInputs, String whatToClassify) {
        this.labelToPredict = whatToClassify;
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
            weights[i] = (float)(Math.random()*2-1);
        }
        return weights;
    }

    /***
     * Run the perceptron on the featureVector and return 0 or 1 for the output category
     * @param featureVector featureVector vector
     * @return 0 or 1 representing the possible output categories or -1 if there's an error
     */
    public int guess(float[] featureVector) {
        float sum = 0;

        for (int i = 0; i < featureVector.length; i++) {
            sum += weights[i]*featureVector[i];
        }

        return activationFunction(sum);
    }

    private int activationFunction(float sum) {
        if (sum > THRESHOLD) {
            return 1;
        } else {
            return 0;
        }
    }

    /***
     * Train the perceptron using the featureVector feature vector and its correct label.
     * Return true if the weights got adjusted (if the prediction was wrong).
     *
     * @param featureVector
     * @param correctLabel
     * @return
     */
    public boolean train(float[] featureVector, String correctLabel) {
        int error = guess(featureVector) - getCorrectGuess(correctLabel);

        if (error == 0) return false;   // no training

        for (int i = 0; i < this.weights.length; i++) {
            weights[i] = weights[i] - error*featureVector[i]*learningRate;
        }

        THRESHOLD = THRESHOLD + error*learningRate;

        return true;
    }

    public float[] getWeights() {
        return weights;
    }

    public String getTargetLabel() {
        return this.labelToPredict;
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
        if (label.equals(this.labelToPredict))
            return 1;
        else
            return 0;
    }

    public float getThreshold() {
        return THRESHOLD;
    }
}