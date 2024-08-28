# screen-forest

## GridSearchCV Best Parameters Output

Given these [observations](data/target_model_data.xlsx), the following could be recommended as default parameters for your use case:

- Bootstrap: True
- Max Depth: None (i.e., no limit), but consider limiting to 20 if model interpretability or overfitting is a concern.
- Min Samples Leaf: 1
- Min Samples Split: 2 (default), with an option to tune between 2 and 5 based on the data size and complexity.
- N Estimators: 200 (this appears to balance accuracy well across different targets)
