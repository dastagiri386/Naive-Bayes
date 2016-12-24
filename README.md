# Naive-Bayes
Implementation of NB and TAN (Tree-augmented Naive Bayes)

## Implementation
For the TAN algorithm :
- The program uses Prim's algorithm to find a maximal spanning tree (but chooses maximal weight edges instead of minimal weight ones). The first variable is chosen as the start vertex. To resolve ties, we use the following preference criteria: (1) prefer edges emanating from variables listed earlier in the input file, (2) if there are multiple maximum weight edges emanating from the first such variable, prefer edges going to variables listed earlier in the input file.
- To root the maximal weight spanning tree, we use the first variable in the input file as the root.

Determines the network structure (in the case of TAN) and estimate the model parameters using the given training set, and then classifies the instances in the test set. The program outputs the following:

- The structure of the Bayes net  (i) the name of the variable, (ii) the names of its parents in the Bayes net (for naive Bayes, this is simply the 'class' variable for every other variable)
- Each instance in the test-set indicating (i) the predicted class, (ii) the actual class, (iii) and the posterior probability of the predicted class 
- The number of the test-set instances that are correctly classified.

## Running the program

```sh
bayes train-set-file test-set-file flag
```
flag is either 't' or 'n' for Naive Bayes or TAN respectively
