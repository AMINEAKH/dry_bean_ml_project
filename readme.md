# 🫘 Dry Bean Classification ML Project

Welcome! This project demonstrates a complete machine learning workflow for classifying dry bean types using the UCI Dry Bean Dataset.

**Key Features:**
- Clean, professional Python project structure
- Data loading, preprocessing, and visualization
- Model training with Random Forest, SVM (RBF), and KNN (optimal k selection)
- Evaluation with classification reports and confusion matrices
- Results, models, and metrics saved for easy review and reproducibility

---

## 📂 Project Structure

```

dry\_bean\_ml\_project/
├── data/
│   └── Dry\_Bean\_Dataset.xlsx
├── models/
│   ├── random\_forest\_model.pkl
│   ├── svm\_model.pkl
│   └── knn\_model.pkl
├── notebooks/
│   └── dry\_bean\_classification.ipynb
├── results/
│   ├── classification\_reports/
│   ├── confusion\_matrices/
│   └── accuracy\_scores.csv
├── main.py
├── requirements.txt
└── README.md

````

---

## 🚀 Getting Started

**1. Clone the repository:**
```bash
git clone https://github.com/AMINEAKH/dry_bean_ml_project.git
cd dry_bean_ml_project
````

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Add the dataset:**
Download [`Dry_Bean_Dataset.xlsx`](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) from UCI and put it in the `data/` folder.

**4. Run the project:**

* **For full code & explanations:**
  Launch Jupyter and run the notebook:

  ```bash
  jupyter notebook notebooks/dry_bean_classification.ipynb
  ```
* **To run as a script:**

  ```bash
  python main.py
  ```

**5. Check results:**

* Trained models: `models/`
* Classification reports: `results/classification_reports/`
* Confusion matrices: `results/confusion_matrices/`
* Accuracy comparison: `results/accuracy_scores.csv`

---

## 📊 Example Results

| Model         | Accuracy |
| ------------- | -------- |
| Random Forest | 0.92     |
| SVM           | 0.91     |
| KNN           | 0.90     |

*(Actual results may vary; see `accuracy_scores.csv`)*

---

## 🧑‍💻 Author

* **Amine Akh** ([AMINEAKH on GitHub](https://github.com/AMINEAKH))

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

You are free to use, modify, and share this code for any purpose — just give credit!

---

## 📑 References

* [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
* [scikit-learn](https://scikit-learn.org/)
* [pandas](https://pandas.pydata.org/)

---

*If you like this project, drop a star ⭐ on the repo!*

