import java.util.*;

// Class representing the KNN (K-Nearest Neighbors) algorithm
public class KNN {
    private final int k;  // Number of neighbors to consider
    private final List<DataPoint> data;  // List to store training data points

    // Constructor to initialize KNN with the value of k
    public KNN(int k) {
        this.k = k;
        this.data = new ArrayList<>();
    }

    // Method to add a single data point to the training data
    public void fit(List<Double> x, String y) {
        this.data.add(new DataPoint(x, y));
    }

    // Method to train the KNN model with a list of data points and corresponding labels
    public void train(List<List<Double>> trainData, List<String> labels) {
        for (int i = 0; i < trainData.size(); i++) {
            fit(trainData.get(i), labels.get(i));
        }
    }

    // Method to predict the label for a given data point
    public String predict(List<Double> x) {
        // Calculate distances between the input data point and all training data points
        List<DistanceLabelPair> distances = new ArrayList<>();
        for (DataPoint point : this.data) {
            double dist = 0;
            for (int j = 0; j < x.size(); j++) {
                dist += Math.pow(x.get(j) - point.getX().get(j), 2);
            }
            distances.add(new DistanceLabelPair(dist, point.getY()));
        }

        // Sort distances in ascending order
        distances.sort(Comparator.comparingDouble(DistanceLabelPair::getDistance));

        // Count votes for each label among the k-nearest neighbors
        Map<String, Double> votes = new HashMap<>();
        for (int i = 0; i < this.k && i < distances.size(); i++) {
            DistanceLabelPair pair = distances.get(i);
            String vote = pair.getLabel();
            double distance = pair.getDistance();
            double voteValue = (distance == 0) ? Double.POSITIVE_INFINITY : 1 / distance;

            votes.put(vote, votes.getOrDefault(vote, 0.0) + voteValue);
        }

        // Determine the predicted label based on majority votes
        String predictedLabel = null;
        double maxVoteValue = Double.MIN_VALUE;
        for (Map.Entry<String, Double> entry : votes.entrySet()) {
            if (entry.getValue() > maxVoteValue) {
                maxVoteValue = entry.getValue();
                predictedLabel = entry.getKey();
            }
        }

        return predictedLabel;
    }

    // Method to predict labels for a list of data points
    public List<String> predictOnData(List<List<Double>> testData) {
        List<String> predictions = new ArrayList<>();
        for (List<Double> dataPoint : testData) {
            predictions.add(predict(dataPoint));
        }
        return predictions;
    }

    // Inner class representing a data point with features (x) and label (y)
    private static class DataPoint {
        private List<Double> x;  // Features of the data point
        private String y;        // Label of the data point

        // Constructor to initialize a data point with features and label
        public DataPoint(List<Double> x, String y) {
            this.x = x;
            this.y = y;
        }

        // Getter method to retrieve the features of the data point
        public List<Double> getX() {
            return x;
        }

        // Getter method to retrieve the label of the data point
        public String getY() {
            return y;
        }
    }

    // Inner class representing a pair of distance and label
    private static class DistanceLabelPair {
        private final double distance;  // Distance between data points
        private final String label;     // Label of the data point

        // Constructor to initialize a distance-label pair
        public DistanceLabelPair(double distance, String label) {
            this.distance = distance;
            this.label = label;
        }

        // Getter method to retrieve the distance
        public double getDistance() {
            return distance;
        }

        // Getter method to retrieve the label
        public String getLabel() {
            return label;
        }
    }
}
