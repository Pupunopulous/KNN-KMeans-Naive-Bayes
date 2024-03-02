import java.text.*;
import java.util.*;

// Generic Pair class to hold key-value pairs
class Pair<K, V> {
    private K key;
    private V value;

    // Constructor to initialize the Pair with key and value
    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }

    // Getter method to retrieve the key
    public K getKey() {
        return key;
    }

    // Getter method to retrieve the value
    public V getValue() {
        return value;
    }
}

// Class representing the KMeans algorithm
public class KMeans {

    // Method to calculate Manhattan distance between two nodes
    public static double manh(Node node1, Node node2) {
        return node1.getPredList().stream()
                .mapToDouble((e1) -> Math.abs(e1 - node2.getPredList().get(node1.getPredList().indexOf(e1))))
                .sum();
    }

    // Method to calculate Euclidean distance squared between two nodes
    public static double e2(Node node1, Node node2) {
        double totalSum = 0;
        for (int idx = 0; idx < node1.getPredList().size(); idx++) {
            totalSum += Math.pow(node1.getPredList().get(idx) - node2.getPredList().get(idx), 2);
        }
        return totalSum;
    }

    // Method to convert input graph data into a list of nodes
    public static List<Node> setNodes(String graphData) {
        List<Node> nodeList = new ArrayList<>();
        String[] splitData = graphData.split("\n");

        for (String line : splitData) {
            line = line.replace(",", " ");
            line = line.trim();
            if (line.isEmpty() || line.charAt(0) == '#') {
                continue;
            }

            String[] tokens = line.split("\\s+");
            Node currentNode = new Node(tokens[tokens.length - 1]);

            for (int i = 0; i < tokens.length - 1; i++) {
                currentNode.addPred(Integer.parseInt(tokens[i]));
            }

            nodeList.add(currentNode);
        }

        return nodeList;
    }

    // Method to convert centroid arguments into a list of nodes
    public static List<Node> setCentroids(List<String> centroidArgs) {
        List<Node> centroidList = new ArrayList<>();

        for (int idx = 0; idx < centroidArgs.size(); idx++) {
            String[] items = centroidArgs.get(idx).replace(",", " ").split("\\s+");
            Node currentNode = new Node("C" + (idx + 1));

            for (String coord : items) {
                currentNode.addPred(Integer.parseInt(coord));
            }

            centroidList.add(currentNode);
        }

        return centroidList;
    }

    // Method to run the KMeans algorithm
    public static void runKMeans(List<Node> dataList, List<Node> centroidList, String distanceFn) {
        int kValue = centroidList.size();
        List<String> output = new ArrayList<>();

        // Iteratively update centroids until convergence
        while (true) {
            Map<Integer, List<Node>> categories = new HashMap<>();

            // Initialize categories
            for (int i = 0; i < kValue; i++) {
                categories.put(i, new ArrayList<>());
            }

            // Assign each node to the closest centroid
            for (Node node : dataList) {
                List<Pair<Integer, Double>> distances = new ArrayList<>();
                for (int idx = 0; idx < kValue; idx++) {
                    if (distanceFn.equals("manh")) distances.add(new Pair<>(idx, manh(node, centroidList.get(idx))));
                    else if (distanceFn.equals("e2")) distances.add(new Pair<>(idx, e2(node, centroidList.get(idx))));
                    else {
                        System.out.println("Incorrect distance function provided for K-Means. Check README for more details.");
                        System.exit(0);
                    }
                }
                distances.sort(Comparator.comparingDouble(Pair::getValue));

                int closestCentroid = distances.get(0).getKey();
                categories.get(closestCentroid).add(node);
            }

            boolean exitFlag = true;

            // Update centroids based on assigned nodes
            for (int i = 0; i < kValue; i++) {
                if (categories.get(i).isEmpty()) {
                    continue;
                }

                Node curCentroid = centroidList.get(i);
                int dimension = curCentroid.getPredList().size();
                List<Double> newCentroid = new ArrayList<>();
                for (int idx = 0; idx < dimension; idx++) {
                    double sum = 0;
                    for (Node node : categories.get(i)) {
                        sum += node.getPredList().get(idx);
                    }
                    newCentroid.add(sum / categories.get(i).size());
                }

                // Check for convergence
                List<Double> diff = new ArrayList<>();
                for (int idx = 0; idx < dimension; idx++) {
                    diff.add(Math.abs(curCentroid.getPredList().get(idx) - newCentroid.get(idx)));
                }

                if (diff.stream().mapToDouble(Double::doubleValue).sum() > 0.00001) {
                    exitFlag = false;
                    curCentroid.setPredList(newCentroid);
                }
            }

            // If converged, print results and exit
            if (exitFlag) {
                for (int i = 0; i < kValue; i++) {
                    System.out.print(centroidList.get(i).getIdentity() + " = {");
                    List<Node> categoryNodes = categories.get(i);
                    for (int j = 0; j < categoryNodes.size(); j++) {
                        System.out.print(categoryNodes.get(j).getIdentity());
                        if (j < categoryNodes.size() - 1) {
                            System.out.print(",");
                        }
                    }
                    System.out.println("}");
                    DecimalFormat df = new DecimalFormat("0.#############");
                    int predSize = centroidList.get(i).getPredList().size();
                    StringBuilder ans = new StringBuilder("([");
                    for (int idx = 0; idx < predSize; idx++) {
                        double val = centroidList.get(i).getPredList().get(idx);
                        ans.append(df.format(val));
                        if (idx != predSize - 1) ans.append(" ");
                    }
                    output.add(ans + "])");
                }
                break;
            }
        }

        // Print the final output
        for (String s : output) System.out.println(s);
    }

    // Method to perform sanity check on input data and centroids
    public static void sanityCheck(List<Node> kMeansList, List<Node> centroidList) {
        int dimension = kMeansList.get(0).getPredList().size();

        // Check dimension consistency for K-Means data
        for (Node node : kMeansList) {
            if (node.getPredList().size() != dimension) {
                System.out.println("Incorrect dimensions for K-Means data input. Check README for more details.");
                System.exit(0);
            }
        }

        // Check dimension consistency for K-Means centroids
        for (Node node : centroidList) {
            if (node.getPredList().size() != dimension) {
                System.out.println("Incorrect dimensions for K-Means centroid arguments. Check README for more details.");
                System.exit(0);
            }
        }
    }
}
