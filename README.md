import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Load dataset
data = pd.read_csv('diabetes.csv')
X = data.drop(columns=['Outcome']).values
y = data['Outcome'].values

# Create and train the model
diabetes_model = DiabetesModel(X, y, hidden_size=64, learning_rate=0.001)
diabetes_model.train(num_epochs=100)

# Evaluate the model
diabetes_model.evaluate()

# Plot results
diabetes_model.plot_results()

# Results
After training the model, you will see the training and validation losses printed every few epochs. The confusion matrix and ROC curve will provide insights into the model's performance on unseen data.

# Visualizations
The following plots are generated:
Confusion Matrix: Displays the true positive, true negative, false positive, and false negative counts.


# Acknowledgements
Kaggle for providing the dataset.
PyTorch documentation for guidance on implementing neural networks.