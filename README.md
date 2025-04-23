# CS771-Project-1-Binary_Classification
CS771-Introduction to Machine Learning (Autumn 2024): Mini-project 1-Report  

This mini-project addresses binary classification using three datasets derived from the same raw data but featuring different input representations: emoticons, deep features, and digit sequences. The goal is to develop models for each dataset individually, optimizing both accuracy and training data usage, while adhering to a 10,000 parameter limit. Additionally, we explore whether combining the datasets through feature merging can enhance model performance. By varying training data sizes and evaluating validation accuracy, the project aims to identify the best-performing models and assess the potential benefits of combining multiple feature representations for improved generalization to unseen data.

## Results : 
* Logistic Regression and SVM emerged as the top-performing models, with SVM showing slightly better accuracy in general. Both models demonstrated the ability to handle the dataset well, especially when a moderate percentage of training data (60%-80%) was used. These models could benefit from further fine-tuning of hyperparameters. 

* Random Forest showed some potential, though its performance was more variable. Hyperpa-rameter tuning (e.g., increasing the number of trees or adjusting tree depth) could help improve its consistency.
  
* Decision Tree and KNN underperformed relative to the other models. These models might benefit from more advanced feature engineering or more specialized hyperparameter optimization.
