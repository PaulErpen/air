# Task 1

## What the f*** is FIRA-22?

The FIRA 22 IR dataset is a collection of documents and queries that was created for the purpose of evaluating and comparing different information retrieval algorithms (IR). The documents in the dataset are news articles from various sources, and the queries are short phrases or sentences that are intended to represent information needs of users.

The specific tasks that the FIRA 22 IR dataset can be used for include, but are not limited to:

Information retrieval: Researchers can use the dataset to develop and evaluate algorithms that can retrieve relevant documents from the dataset given a query.

Query expansion: The dataset can be used to develop and evaluate algorithms that can automatically expand the query with additional terms or concepts that are related to the original query.

Document clustering: Researchers can use the dataset to develop and evaluate algorithms that can group similar documents together based on their content.

Information extraction: The dataset can be used to develop and evaluate algorithms that can extract specific pieces of information (such as named entities or events) from the documents in the dataset.

Overall, the FIRA 22 IR dataset is a valuable resource for researchers who are interested in developing and evaluating information retrieval algorithms.

## What the f*** are Qrels?

"Qrels" is a term commonly used in the field of information retrieval and it stands for "query relevance judgments". Qrels are essentially a list of relevance judgments for a set of queries and documents, similar to the judgments provided in the FIRA 22 IR dataset.

Qrels are typically used to evaluate the effectiveness of information retrieval algorithms. Specifically, given a set of qrels, researchers can use them to calculate various evaluation metrics such as precision, recall, and F1 score, which provide a quantitative measure of the performance of their algorithms.

In the case of the FIRA 22 IR dataset, qrels are provided for each query in the dataset, specifying which documents are relevant to that query according to the human judges. These qrels are usually represented in a standardized format, such as the TREC (Text Retrieval Conference) format, which is a widely used format for qrels in the information retrieval community.

## What alternatives do we have for majority voting?

There are several alternatives to majority voting for aggregating ordinal ratings in the context of information retrieval. Here are a few common methods:

Average rating: This method involves calculating the average rating across all judges for a given query-answer pair. This can be a simple and effective way to aggregate ratings.

Weighted average rating: Similar to the average rating method, but allows for different judges to be given different weights based on their expertise or reliability. For example, more weight could be given to ratings from judges who have a track record of providing more accurate ratings.

Borda count: This method involves assigning points to each rating, based on its position in the ranking (e.g., the highest rating gets n points, the second highest gets n-1 points, and so on), and then summing up the points across all judges for a given query-answer pair. This method can be effective for dealing with a small number of judges and can help to mitigate the effects of outliers.

Kendall's W coefficient: This is a measure of agreement among judges that takes into account the ordinal nature of the ratings. It calculates the degree of agreement between judges based on how often each pair of judges ranks the query-answer pairs in the same order. This method can be useful for identifying disagreements among judges and can be used to adjust the ratings accordingly.

Multidimensional scaling: This is a more advanced technique that involves mapping the ordinal ratings to a low-dimensional space and then calculating distances between pairs of query-answer pairs based on the mapped ratings. This can help to identify clusters of similar query-answer pairs and can be useful for identifying patterns in the ratings that may not be apparent from the raw data.

Each of these methods has its own strengths and weaknesses, and the choice of method will depend on the specific context and the goals of the analysis.

## Elaboration on Kendall's W coefficient

Kendall's W coefficient is a measure of agreement among judges that takes into account the ordinal nature of the ratings. It is a variation of Kendall's tau-b correlation coefficient, which is a measure of association between two ranked lists. The coefficient ranges from 0 to 1, with higher values indicating greater agreement among judges.

To calculate Kendall's W, you first need to have ratings from at least three judges for each query-answer pair. The ratings should be in ordinal scale, meaning that each judge assigns a score to each query-answer pair, typically on a scale from 1 (worst match) to N (best match), where N is the number of possible scores. Then, the following steps are taken:

Calculate the concordance matrix: For each pair of judges, calculate the number of query-answer pairs for which the two judges gave the same rating (concordant) and the number of pairs for which they gave different ratings (discordant). This results in a matrix where each element represents the number of concordant ratings between the two judges.

Calculate the average number of concordant ratings: For each query-answer pair, calculate the average number of concordant ratings across all pairs of judges. This gives an estimate of the overall agreement among the judges for that pair.

Calculate the average number of concordant ratings expected by chance: This step involves calculating the expected number of concordant ratings if the judges were assigning ratings at random. This is based on the proportion of concordant ratings expected by chance, which is a function of the number of possible ratings and the number of judges. The expected number of concordant ratings is then calculated as the sum of the products of the expected proportion of concordant ratings and the number of pairs of judges.

Calculate Kendall's W: Finally, Kendall's W is calculated as the difference between the average number of concordant ratings and the average number of concordant ratings expected by chance, divided by the maximum possible difference between the two (i.e., the total number of pairs of judges minus one). This gives a measure of the degree of agreement among the judges that takes into account the expected agreement by chance.

Kendall's W coefficient can be a useful tool for identifying pairs of judges who are consistently in agreement or disagreement, as well as for identifying query-answer pairs that are consistently rated highly or poorly by the judges. It can also be used to adjust the ratings for individual judges based on their level of agreement with the others. However, it is important to note that Kendall's W only takes into account the ordinal nature of the ratings, and does not consider the actual scores assigned by the judges. As such, it may not be the best method to use if the scores themselves are important for the analysis.

## In the context of neural networks, what does a backward pass and gradient update per batch refer to?

In neural networks, the backward pass (also known as backpropagation) is a process that is used to compute the gradients of the model parameters with respect to the loss function. During training, the forward pass is used to compute the output of the model given the input data, and then the backward pass is used to compute the gradients of the parameters with respect to the loss function.

The gradient update per batch refers to the process of updating the model parameters using the gradients computed in the backward pass. In batch training, the data is divided into small batches, and the gradients are computed and averaged over each batch. The model parameters are then updated based on the average gradients of the batch.

The process of updating the model parameters involves multiplying the gradients by a learning rate and subtracting the result from the current parameter values. The learning rate controls the step size of the update and is typically a small positive number.

The backward pass and gradient update per batch are repeated iteratively over multiple epochs until the model converges to a satisfactory level of performance on the training data. The number of iterations required to converge depends on various factors such as the size and complexity of the model, the size of the training dataset, and the choice of hyperparameters.