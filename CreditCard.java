import java.io.*;
import java.util.*;

public class CreditCard
{
  private Map<Integer,String> headers = new TreeMap<>();
  private double[] weights;
  private String labelName = "Class";
  private int randomSeed = -1;
  private int sets = 20;
  private int folds = 5;
  private double threshold = 0.998;
  private Random rand;

  public static void main(String[] args)
  {
    CreditCard model = new CreditCard();

    model.labelName = System.getProperty("Label", model.labelName);
    model.randomSeed = Integer.parseInt(System.getProperty("Seed", model.randomSeed + ""));
    model.sets = Integer.parseInt(System.getProperty("sets", model.sets + ""));
    model.folds = Integer.parseInt(System.getProperty("folds", model.folds + ""));
    model.threshold = Double.parseDouble(System.getProperty("threshold", model.threshold + ""));

  model.rand = model.randomSeed >= 0 ? new Random(model.randomSeed) : new Random();

  model.modeller(args);
  }

  public void modeller(String... fn)
  {
    List<Map<String,Double>> allData = load(fn, "Time");

    System.out.println("Rows Loaded : " + allData.size());

    // Amount should be normalised
    allData = normalise(allData, "Amount");

    // Data is more convenient as arrays
    double[] labels = new double[allData.size()];
    double[][] atts = new double[headers.size() - 1][allData.size()];
    String[] heads = new String[headers.size() - 1];

    flatten(allData, heads, labels, atts, true);

    // How statistically significant are features
    Map<Integer,String> newHeads = significant(atts, labels, heads, 0.5, 1.0);

    keep(allData, newHeads.values().toArray(new String[0]));

    headers = newHeads;

    // split data up for test training and testing
    List<Map<String,Double>>[] landT = apportion(allData, 0.8);
    List<Map<String,Double>> data = landT[0];
    List<Map<String,Double>> test = landT[1];

    System.out.println("Train : " + data.size());
    System.out.println("Test  : " + test.size());

    // reset array sizes for training on subset of features
    labels = new double[data.size()];
    atts = new double[headers.size() - 1][data.size()];
    heads = new String[headers.size() - 1];

    double[] bestWeights = null;
    double lastF1 = 0.0;

    // As data unbalanced, use several random subsets
    for (int i = 0; i != sets; i++)
    {
      // Get a subsample of the data (all frauds, and balanced set of non-fraud)
      List<Map<String,Double>> sample = subSample(data, 0.0, 0.002);

      // N fold validation
      for (int j = 0; j != folds; j++)
      {
        // Split up for training and validation data folds
        List<Map<String,Double>>[] data2 = apportion(sample, 0.8);

        List<Map<String,Double>> train = data2[0];
        List<Map<String,Double>> validate = data2[1];

        flatten(train, heads, labels, atts, true);

        weights = train(0.0001, 10000, atts, labels);

        double f1 = evaluate(validate);

        if (f1 > lastF1)
        {
          lastF1 = f1;
          bestWeights = weights;
        }
      }
    }

    System.out.println();
    System.out.println("Best F1 = " + lastF1);

    weights = bestWeights;

    System.out.println();
    double f1all = evaluate(test, true);
  }

  private double evaluate(List<Map<String,Double>> valData)
  {
    return evaluate(valData, -1.0, false);
  }

  private double evaluate(List<Map<String,Double>> valData, double label)
  {
    return evaluate(valData, label, false);
  }

  private double evaluate(List<Map<String,Double>> valData, boolean show)
  {
    return evaluate(valData, -1.0, show);
  }

  private double evaluate(List<Map<String,Double>> valData, double label, boolean show)
  {
    double[] cLabels = new double[valData.size()];
    double[][] cAtts = new double[valData.size()][headers.size() - 1];
    String[] cHeads = new String[headers.size() - 1];

    flatten(valData, cHeads, cLabels, cAtts, false);

    int count = 0;
    double[][] confusionMatrix = new double[2][2];
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;

    for (int k = 0; k != cLabels.length; k++)
    {
      double thisLabel = valData.get(k).get(labelName);

      if (label < 0.0 || thisLabel == label)
      {
        double predicted = classify(weights, cAtts[k], false);

        if (predicted > threshold)
        {
          if (thisLabel == 1.0)
          {
            accuracy += 1;
            precision += 1;
            recall += 1;
            confusionMatrix[0][0] += 1;
          }
          else
          {
            confusionMatrix[1][0] += 1;
          }
        }
        else
        {
          if (valData.get(k).get(labelName) == 1.0)
          {
            confusionMatrix[0][1] += 1;
          }
          else
          {
            accuracy += 1;
            confusionMatrix[1][1] += 1;
          }
        }

        count++;
      }
    }

    accuracy /= count;
    precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
    recall /= confusionMatrix[0][0] + confusionMatrix[0][1];

    double f1 = 2.0 * precision * recall / (precision + recall);

    if (show)
    {
      System.out.println();
      System.out.println(Arrays.deepToString(confusionMatrix));
      System.out.printf("Accuracy  : %.1f %%\n", accuracy * 100);
      System.out.printf("Precision : %.1f %%\n", precision * 100);
      System.out.printf("Recall    : %.1f %%\n", recall * 100);
      System.out.printf("F1        : %.1f %%\n", f1 * 100);
    }

    return f1;
  }

  private static double pearsonCorrelation(double[] xn, double[] yn)
  {
    return pearsonCorrelation(xn, yn, null, 0.0);
  }

  private static double pearsonCorrelation(double[] xn, double[] yn, double[] labels, double label)
  {
    long n = xn.length;        
    double xSum = 0;
    double ySum = 0;
    double xySum = 0;
    double xxSum = 0;
    double yySum = 0;
    
    for (int i = 0; i != n; i++)
    {
      if (labels == null || labels[i] == label)
      {
        double x = xn[i];
        double y = yn[i];

        if (x == 0.0 && label == 0.0)
          x = -1.0;
        if (y == 0.0 && label == 0.0)
          y = -1.0;

        xSum += x;
        ySum += y;
        xySum += x * y;
        xxSum += x * x;
        yySum += y * y;
      }
    }

    double xd = Math.sqrt(n * xxSum - xSum * xSum);
    double yd = Math.sqrt(n * yySum - ySum * ySum);

    if (xd == 0.0)
      xd = 1.0;
    if (yd == 0.0)
      yd = 1.0;
    
    return (n * xySum - xSum * ySum) / (xd * yd);
  }

  public static double[] train(double rate, int its, double[][] data, double[] labels)
  {
    double[] weights = new double[data.length - 1];

    for (int n = 0; n < its; n++)
    {
      for (int i = 0; i < data.length; i++)
      {
        double[] x = data[i];
        double predicted = classify(weights, x);
        double label = labels[i];

        for (int j = 0; j < weights.length; j++)
        {
          // Conspicuous by it's absence is regularisation!
          weights[j] = weights[j] + rate * (label - predicted) * x[j];
        }
      }
    }

    return weights;
  }

  private static double classify(double[] weights, double[] x)
  {
      return classify(weights, x, false);
  }

  private static double classify(double[] weights, double[] x, boolean debug)
  {
    double logit = 0.0;

    for (int i = 0; i < weights.length; i++)
      logit += weights[i] * x[i];

    if (debug)
      System.out.println((1.0 / (1.0 + Math.exp(-logit))));

    // sigmoid
    return 1.0 / (1.0 + Math.exp(-logit));
  }

  @SuppressWarnings("unchecked")
  public List<Map<String,Double>>[] apportion(Collection<Map<String,Double>> arr, double ratio)
  {
    List<Map<String,Double>>[] split = (ArrayList<Map<String,Double>>[]) new ArrayList[2];
    int l = arr.size();
    int n = (int) (l * ratio);
    List<Map<String,Double>> one = new ArrayList<>(n + 1);
    List<Map<String,Double>> two = new ArrayList<>(l - n + 1);
    int g = 0;

    split[0] = one;
    split[1] = two;

    for (Map<String,Double> i : arr)
    {
      if (rand.nextDouble() < ratio && g <= n)
      {
        one.add(i);

        g++;
      }
      else
        two.add(i);
    }

    return split;
  }

  private List<Map<String,Double>> subSample(Collection<Map<String,Double>> arr, double label, double sampleSize)
  {
    List<Map<String,Double>> sample = new ArrayList<Map<String,Double>>((int) (arr.size() * sampleSize * 2.0));
    int in = 0;
    int out = 0;

    for (Map<String,Double> ents : arr)
    {
      double clazz = ents.get(labelName);

      if (clazz == label)
      {
        if (rand.nextDouble() < sampleSize)
        {
          sample.add(ents);

          in++;
        }
      }
      else
      {
        sample.add(ents);

        out++;
      }
    }

    return sample;
  }

  private void flatten(List<Map<String,Double>> data, String[] hds, double[] labels, double[][] atts, boolean byRow)
  {
    int row = 0;

    for (Map<String,Double> ents : data)
    {
      flatten(row, ents, hds, labels, atts, byRow);

      row++;
    }
  }

  private void flatten(Map<String,Double> data, String[] hds, double[] labels, double[][] atts, boolean byRow)
  {
    flatten(0, data, hds, labels, atts, byRow);
  }

  private void flatten(int row, Map<String,Double> data, String[] hds, double[] labels, double[][] atts, boolean byRow)
  {
    int i = 0;

    for (Map.Entry<String,Double> e : data.entrySet())
    {
      String key = e.getKey();

      if (labelName.equals(key))
      {
        labels[row] = e.getValue();
      }
      else
      {
        if (row == 0)
          hds[i] = key;

        if (byRow)
          atts[i++][row] = e.getValue();
        else
          atts[row][i++] = e.getValue();
      }
    }
  }

  private Map<Integer,String> significant(double[][] atts, double[] labels, String[] heads, double threshold)
  {
    return significant(atts, labels, heads, threshold, -1.0);
  }

private Map<Integer,String> significant(double[][] atts, double[] labels, String[] heads, double threshold, double label)
  {
    List<String> hds = new ArrayList<>();
    int i = 0;

    for (double[] at : atts)
    {
      double corr = pearsonCorrelation(at, labels, label >= 0.0 ? labels : null, 1.0);

      if (Math.abs(corr) > threshold)
      {
        hds.add(heads[i]);
      }

      i++;
    }

    hds.add(labelName);  // Add in the Label

    Map<Integer,String> hm = new TreeMap<>();

    for (int j = 0; j != hds.size(); j++)
      hm.put(j, hds.get(j));

    return hm;
  }

  private void remove(List<Map<String,Double>> data, String... hds)
  {
    for (Map<String,Double> ents : data)
    {
      for (String hdr : hds)
        ents.remove(hdr);
    }
  }

  private void keep(List<Map<String,Double>> data, String... hds)
  {
    List<String> rms = new ArrayList<>(headers.values());

    rms.removeAll(Arrays.asList(hds));

    remove(data, rms.toArray(new String[0]));
  }

  private static List<Map<String,Double>> normalise(List<Map<String,Double>> data, String... hds)
  {
    Map<String, Double> sums = new HashMap<>();

    for (Map<String,Double> ents : data)
    {
      for (String hdr : hds)
      {
        double amount = ents.get(hdr);

        if (sums.containsKey(hdr))
          sums.replace(hdr, amount + sums.get(hdr));
        else
          sums.put(hdr, amount);
      }
    }

    for (Map<String,Double> ents : data)
    {
      for (String hdr : hds)
      {
        double amount = ents.get(hdr);
        double total = sums.get(hdr);

        ents.replace(hdr, amount / total);
      }
    }

    return data;
  }

  private List<Map<String,Double>> load(String[] fns, String... excludes)
  {
    List<Map<String,Double>> data = new LinkedList<>();

    Set<String> exHead = new HashSet<>();
    Set<Integer> exPos = new HashSet<>();

    for (String hd : excludes)
      exHead.add(hd);

    for (String fn : fns)
    {
      System.out.println("Loading : " + fn);

      String cvsSplitBy = fn.endsWith(".txt") ? "\t" : ",";

      try (BufferedReader br = new BufferedReader(new FileReader(fn)))
      {
        boolean first = true;
        String line;

        while ((line = br.readLine()) != null)
        {
          if (first)
          {
            int i = 0;

            for (String hdr : line.split(cvsSplitBy))
            {
              if (hdr.startsWith("\"") && hdr.endsWith("\"") || hdr.startsWith("'") && hdr.endsWith("'"))
                hdr = hdr.substring(1, hdr.length() - 1);

              if (exHead.contains(hdr))
              {
                exPos.add(i);
              }
              else
              {
                headers.put(i, hdr);
              }

              i++;
            }

            first = false;

            continue;
          }

          // use comma as separator
          String[] items = line.split(cvsSplitBy);
          Map<String,Double> its = new TreeMap<>();

          try
          {
            int i = 0;

            for (String it : line.split(cvsSplitBy))
            {
              if (it.startsWith("\"") && it.endsWith("\"") || it.startsWith("'") && it.endsWith("'"))
                it = it.substring(1, it.length() - 1);

              try
              {
                if (! exPos.contains(i))
                  its.put(headers.get(i), Double.parseDouble(it));
              }
              catch (Exception e)
              {
                System.err.println(e);
              }

              i++;
            }

            data.add(its);
          }
          catch (ArrayIndexOutOfBoundsException ae)
          {
            System.err.println("Bad Record : " + line);
          }
        }
      }
      catch (Exception e)
      {
        e.printStackTrace();
      }
    }

    return data;
  }
}
