import java.util.*;

// Class representing an evaluator for label metrics
public class Evaluator {

    // Inner class to store label metrics (correct predictions, total predictions, true occurrences)
    public static class LabelMetrics {
        private int correct;       // Number of correct predictions
        private int predicted;     // Total number of predictions
        private int trueCount;     // Total occurrences of the true label

        // Constructor to initialize label metrics
        public LabelMetrics(int correct, int predicted, int trueCount) {
            this.correct = correct;
            this.predicted = predicted;
            this.trueCount = trueCount;
        }

        // Getter method to retrieve the number of correct predictions
        public int getCorrect() {
            return correct;
        }

        // Getter method to retrieve the total number of predictions
        public int getPredicted() {
            return predicted;
        }

        // Getter method to retrieve the total occurrences of the true label
        public int getTrueCount() {
            return trueCount;
        }
    }

    // Method to evaluate label metrics based on actual and predicted labels
    public static Map<String, LabelMetrics> evaluateMetrics(List<String> actualLabels, List<String> predictedLabels) {
        Map<String, LabelMetrics> metricsDict = new HashMap<>();

        // Iterate through actual and predicted labels to update metrics
        for (int i = 0; i < actualLabels.size(); i++) {
            String actual = actualLabels.get(i);
            String predicted = predictedLabels.get(i);

            // Increase true count for the actual label
            metricsIncrease(metricsDict, actual, "true");

            // If actual label equals predicted label, increase correct count for the actual label
            if (actual.equals(predicted)) {
                metricsIncrease(metricsDict, actual, "correct");
            }

            // Increase predicted count for the predicted label
            metricsIncrease(metricsDict, predicted, "predicted");
        }

        return metricsDict;
    }

    // Private method to increase the corresponding metric for a given label
    private static void metricsIncrease(Map<String, LabelMetrics> metricsDict, String label, String type) {
        // If the label is not in the metrics dictionary, add it with initial metrics
        if (!metricsDict.containsKey(label)) {
            metricsDict.put(label, new LabelMetrics(0, 0, 0));
        }

        // Retrieve label metrics for the given label
        LabelMetrics labelMetrics = metricsDict.get(label);

        // Increase the corresponding metric based on the specified type
        switch (type) {
            case "correct" -> labelMetrics.correct++;
            case "predicted" -> labelMetrics.predicted++;
            case "true" -> labelMetrics.trueCount++;
        }
    }
}
