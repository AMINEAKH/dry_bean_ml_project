
## 🚀 How to Run

1. **Clone or Download** this repository.
2. **Install dependencies** :
    ```
    pip install -r requirements.txt
    ```
3. **Run the notebook** for a step-by-step explanation and visuals:
    ```
    jupyter notebook notebooks/dry_bean_classification.ipynb
    ```
   _or just run the script to generate outputs in one shot:_
    ```
    python main.py
    ```
4. **Check the `results/` folder** for model evaluation results.

## 📝 What Each Folder/File Is For

- **data/**: The original dataset file (.xlsx).
- **notebooks/**: Jupyter notebook for detailed EDA, model building, and explanation.
- **models/**: Saved, trained models (pickled).
- **results/classification_reports/**: Text reports with metrics for each model.
- **results/confusion_matrices/**: PNG images of each confusion matrix.
- **results/accuracy_scores.csv**: Table comparing all model accuracies.
- **main.py**: Full workflow in script form (no markdown/explanation).
- **requirements.txt**: All required Python libraries.
- **README.md**: This file.

## 🛠️ Features

- Data loading, preprocessing, and train/test split
- Automatic label encoding and feature scaling
- Model training: Random Forest, SVM (RBF), KNN (with auto-best k)
- Saving trained models for later use
- Classification reports and confusion matrices (text & image)
- Accuracy comparison (CSV)
- Fully reproducible via notebook or script

## Credits

- Dataset: [UCI Machine Learning Repository - Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- Author: Mohamed Amine Akhdaich

---

Feel free to fork, star, or build on this project!
