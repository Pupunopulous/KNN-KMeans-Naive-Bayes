import java.io.*;
import java.util.*;

// Class representing a node in a graph
class Node {
    private final String identity;   // Unique identifier for the node
    private List<Double> predList;   // List to store predictions associated with the node

    // Constructor to initialize the node with an identity
    public Node(String identity) {
        this.identity = identity;
        this.predList = new ArrayList<>();
    }

    // Getter method to retrieve the identity of the node
    public String getIdentity() {
        return identity;
    }

    // Getter method to retrieve the prediction list of the node
    public List<Double> getPredList() {
        return predList;
    }

    // Setter method to set the prediction list of the node
    public void setPredList(List<Double> predList) {
        this.predList = predList;
    }

    // Method to add a prediction to the node's prediction list
    public void addPred(double pred) {
        predList.add(pred);
    }
}

// Main class for the program
public class learn {

    // Lists to store training and testing data along with corresponding labels
    private static final List<List<Double>> trainData = new ArrayList<>();
    private static final List<String> trainLabels = new ArrayList<>();
    private static final List<List<Double>> testData = new ArrayList<>();
    private static final List<String> testLabels = new ArrayList<>();

    // Method to print the comparison between actual and predicted labels
    public static void printPredictionComparisons(List<String> testLabels, List<String> predictions) {
        for (int i = 0; i < testLabels.size(); i++) {
            System.out.println("want=" + testLabels.get(i) + " got=" + predictions.get(i));
        }
    }

    // Method to print evaluation metrics for different labels
    public static void printMetrics(Map<String, Evaluator.LabelMetrics> metricsDict) {
        for (String label : metricsDict.keySet()) {
            Evaluator.LabelMetrics labelMetrics = metricsDict.get(label);
            System.out.println("Label=" + label +
                    " Precision=" + labelMetrics.getCorrect() + "/" + labelMetrics.getPredicted() +
                    " Recall=" + labelMetrics.getCorrect() + "/" + labelMetrics.getTrueCount());
        }
    }

    // Method to read CSV file and populate training or testing data and labels
    private static void readCSV(String filename, boolean isTrain) {
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            br.lines().forEach(line -> {
                if (!line.isEmpty()) {
                    String[] values = line.split(",");
                    List<Double> row = new ArrayList<>();

                    // Label is in the last column
                    int lastIndex = values.length - 1;

                    // Storing the label
                    if (isTrain) trainLabels.add(values[lastIndex]);
                    else testLabels.add(values[lastIndex]);

                    // Storing the numbers in other columns
                    for (int i = 0; i < lastIndex; i++) {
                        row.add(Double.parseDouble(values[i]));
                    }
                    if (isTrain) trainData.add(row);
                    else testData.add(row);
                }
            });
        } catch (IOException e) {
            System.out.println("One or more argument file(s) not found. Terminating program.");
            System.exit(1);
        }
    }

    // Method to read KMeans CSV file and return its content as a string
    public static String readKMeansCSV(String filename) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            StringBuilder inputData = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                inputData.append(line).append("\n");
            }
            return inputData.toString();
        } catch (IOException e) {
            System.out.println("One or more argument file(s) not found. Terminating program.");
            System.exit(0);
        }
        return null;
    }

    // Driver method of the program
    public static void main(String[] args) {
        // Check if the correct number of command line arguments is provided
        if (args.length < 4) {
            System.out.println("Incorrect number of arguments passed. Check README for more details.");
            System.exit(1);
        }

        // Variables to store file names, k value, c value, distance function, centroids, and verbosity flag
        String trainFile = "";
        String testFile = "";
        int k = 0;
        double c = 0;
        String distanceFn = "";
        List<String> centroids = new ArrayList<>();
        boolean verbose = false;

        // Parse command line arguments
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-train" -> trainFile = args[++i];
                case "-test" -> testFile = args[++i];
                case "-k" -> k = Integer.parseInt(args[++i]);
                case "-c" -> c = Double.parseDouble(args[++i]);
                case "-d" -> distanceFn = args[++i];
                case "-v", "-verbose" -> verbose = true;
                default -> {
                    if (args[i].contains(",")) centroids.add(args[i]);
                    else {
                        System.out.println("One or more incorrect arguments passed. Check README for more details.");
                        System.exit(0);
                    }
                }
            }
        }

        // Check if K-Means is specified
        if (!distanceFn.equals("")) {
            // Check if centroids are provided for K-Means
            if (centroids.isEmpty()) {
                System.out.println("Incorrect centroids provided for K-Means. Check README for more details.");
                System.exit(0);
            }
            // Read KMeans CSV file, set nodes and centroids, perform sanity check, and run K-Means
            String kmeansData = readKMeansCSV(trainFile);
            List<Node> kmeansNodes = KMeans.setNodes(kmeansData);
            List<Node> centroidList = KMeans.setCentroids(centroids);
            KMeans.sanityCheck(kmeansNodes, centroidList);
            KMeans.runKMeans(kmeansNodes, centroidList, distanceFn);
        } else {
            // Check validity of k, c values
            if (k < 0) {
                System.out.println("Error: Number of nearest neighbours \"K\" must be >= 0.");
                System.exit(1);
            }
            if (c < 0) {
                System.out.println("Error: Laplacian correction \"C\" must be >= 0.");
                System.exit(1);
            }
            if (k > 0 && c > 0) {
                System.out.println("Error: cannot use both \"K\" and \"C\" in the same algorithm.");
                System.exit(1);
            }

            // Read training and testing data
            readCSV(trainFile, true);
            readCSV(testFile, false);

            // Perform KNN or Naive Bayes based on the specified algorithm
            if (k > 0) {
                KNN knn = new KNN(k);
                knn.train(trainData, trainLabels);
                List<String> predictions = knn.predictOnData(testData);
                if (verbose) {
                    printPredictionComparisons(testLabels, predictions);
                }
                Map<String, Evaluator.LabelMetrics> metricsDict = Evaluator.evaluateMetrics(testLabels, predictions);
                printMetrics(metricsDict);
            } else {
                NaiveBayes naiveBayes = new NaiveBayes(c, verbose);
                naiveBayes.train(trainData, trainLabels);
                List<String> predictions = naiveBayes.predictOnData(testData, testLabels);
                Map<String, Evaluator.LabelMetrics> metricsDict = Evaluator.evaluateMetrics(testLabels, predictions);
                Map<String, Evaluator.LabelMetrics> sortedMetrics = new TreeMap<>(metricsDict);
                printMetrics(sortedMetrics);
            }
        }
    }
}
