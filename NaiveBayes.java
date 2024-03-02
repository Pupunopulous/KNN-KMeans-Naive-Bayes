import java.text.*;
import java.util.*;

public class NaiveBayes {
    // Class to perform Naive Bayes classification with Laplacian smoothing.

    // Hyperparameters and data structures for training and storing probabilities
    private double c;  // Laplacian smoothing parameter
    private boolean verbose;  // Flag for verbose output
    private List<Set<String>> valueSet = new ArrayList<>();  // Set of unique values for each feature/column
    private Map<String, Double> pureProbabilities = new TreeMap<>();  // Prior probabilities of labels
    private Map<String, String> pureProbabilitiesDesc = new TreeMap<>();  // String representation of pure probabilities
    private Map<Triplet, Double> condProbabilities = new HashMap<>();  // Conditional probabilities of features given labels
    private Map<Triplet, String> condProbabilitiesDesc = new HashMap<>();  // String representation of conditional probabilities

    // Decimal format for output precision
    private final DecimalFormat df = new DecimalFormat("0.#############");

    // Constructor to initialize hyperparameters
    public NaiveBayes(double c, boolean verbose) {
        this.c = c;
        this.verbose = verbose;
    }

    // Method to train the Naive Bayes classifier
    public void train(List<List<Double>> trainData, List<String> labels) {
        int totalNum = trainData.size();

        // Collect unique values for each feature/column
        for (int i = 0; i < trainData.get(0).size(); i++) {
            Set<String> valueSetColumn = new HashSet<>();
            for (List<Double> row : trainData) {
                valueSetColumn.add(String.valueOf(row.get(i)));
            }
            valueSet.add(valueSetColumn);
        }
        valueSet.add(new TreeSet<>(labels));  // Add labels to value set

        // Count occurrences for pure and conditional probabilities
        Map<String, Integer> pureCountDict = new TreeMap<>();
        Map<Triplet, Integer> condCountDict = new HashMap<>();
        for (int i = 0; i < totalNum; i++) {
            List<Double> rowX = trainData.get(i);
            String rowY = labels.get(i);
            for (int j = 0; j < rowX.size(); j++) {
                double xCol = rowX.get(j);
                Triplet key = new Triplet(String.valueOf(xCol), j, rowY);
                condCountDict.put(key, condCountDict.getOrDefault(key, 0) + 1);
            }
            pureCountDict.put(rowY, pureCountDict.getOrDefault(rowY, 0) + 1);
        }

        // Calculate pure and conditional probabilities
        for (String label : valueSet.get(valueSet.size() - 1)) {
            int count = pureCountDict.getOrDefault(label, 0);
            pureProbabilities.put(label, (double) count / totalNum);
            pureProbabilitiesDesc.put(label, count + " / " + totalNum);

            for (int i = 0; i < valueSet.size() - 1; i++) {
                int xColDom = valueSet.get(i).size();
                for (String xCol : valueSet.get(i)) {
                    Triplet key = new Triplet(xCol, i, label);
                    int countCond = condCountDict.getOrDefault(key, 0);
                    double probability = (countCond + c) / (pureCountDict.get(label) + c * xColDom);
                    condProbabilities.put(key, probability);
                    condProbabilitiesDesc.put(key, df.format(countCond + c) + " / " +
                            df.format(pureCountDict.get(label) + c * xColDom));
                }
            }
        }
    }

    // Method to calculate the probability of a label given a set of features
    public double calculateYProb(List<String> x, String y) {
        if (!valueSet.get(valueSet.size() - 1).contains(y)) {
            // Warning if the label is not in the training label set
            System.out.println("Warning: Label " + df.format(Double.parseDouble(y)) +
                    " does not exist in training label set");
            return 0;
        }

        if (verbose) {
            System.out.println("P(C=" + y + ") = [" + pureProbabilitiesDesc.get(y) + "]");
        }

        double prob = pureProbabilities.get(y);
        if (x.size() > valueSet.size() - 1) {
            // Error if the input features have more dimensions than training data
            System.out.println("Error: X (" + x + ") has more features than training data.");
            System.exit(1);
        }

        // Calculate the conditional probabilities for each feature
        for (int i = 0; i < x.size(); i++) {
            String xCol = x.get(i);

            if (!valueSet.get(i).contains(xCol)) {
                // Warning if the feature value is not in the training set
                System.out.println("Warning: X value " + df.format(Double.parseDouble(xCol)) +
                        " for column #" + (i + 1) + " not in training set.");
                return 0;
            }

            Triplet key = new Triplet(xCol, i, y);
            if (verbose) {
                System.out.println("P(A" + df.format(i) + "=" + df.format(Double.parseDouble(xCol)) +
                        " | C=" + y + ") = " + condProbabilitiesDesc.get(key));
            }

            prob *= condProbabilities.get(key);
        }
        return prob;
    }

    // Method to predict the label for a given set of features
    public String predict(List<String> x, String y) {
        List<String> labels = new ArrayList<>();
        List<Double> probs = new ArrayList<>();

        // Calculate probabilities for each label
        for (String value : valueSet.get(valueSet.size() - 1)) {
            double prob = calculateYProb(x, value);
            labels.add(value);
            probs.add(prob);
        }

        // Print probabilities and determine the predicted label
        for (int i = 0; i < labels.size(); i++) {
            String label = labels.get(i);
            double prob = probs.get(i);
            if (verbose) {
                System.out.println("NB(C=" + label + ") = " + String.format("%.6f", prob));
            }
        }

        double maxProb = Collections.max(probs);
        int maxIndex = probs.indexOf(maxProb);

        if (verbose) {
            if (labels.get(maxIndex).equals(y))
                System.out.println("match: \"" + labels.get(maxIndex) + "\"");
            else
                System.out.println("fail: got \"" + labels.get(maxIndex) + "\" != want \"" + y + "\"");
        }

        return labels.get(maxIndex);
    }

    // Method to predict labels for a set of data points
    public List<String> predictOnData(List<List<Double>> testData, List<String> testLabels) {
        List<String> predictions = new ArrayList<>();
        for (int i = 0; i < testData.size(); i++) {
            List<Double> dataPoint = testData.get(i);
            List<String> stringDataPoint = new ArrayList<>();
            for (Double value : dataPoint) {
                stringDataPoint.add(String.valueOf(value));
            }
            predictions.add(predict(stringDataPoint, testLabels.get(i)));
        }
        return predictions;
    }

    // Class to represent a triplet (feature value, feature index, label)
    private static class Triplet {
        private final String first;
        private final int second;
        private final String third;

        public Triplet(String first, int second, String third) {
            this.first = first;
            this.second = second;
            this.third = third;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Triplet triplet = (Triplet) o;
            return second == triplet.second &&
                    Objects.equals(first, triplet.first) &&
                    Objects.equals(third, triplet.third);
        }

        @Override
        public int hashCode() {
            return Objects.hash(first, second, third);
        }
    }
}
