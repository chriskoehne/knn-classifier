# KNN Classifier (Transfer Learning)
A transfer learning KNN classifier used to try and assign resources for students who would use them the most effectively.

- There are only enough resources for the <i>best</i> 10% of positive performing students, assuming a 50/50 split between positive and negative performing
- Non-binary data is normalized to a 0-1 scale
- Nominal data is to binary numbers (0001 for the first option, 0010 for the second, and so on) and then normalized.
- Euclidian distance was found to be the most efficient method for calculating the distance between neighbors (as opposed to Lp – Norm distance with p = the number of categories)
- A final k value of 9 is chosen for the transfer learning

The final classifier is in knn_classifier.py

Descriptions of the dataset's attributes are found in Data Descriptions.txt
