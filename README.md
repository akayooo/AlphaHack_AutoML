# AutoML Pipeline for Binary Classification

This repository contains an automated machine learning (AutoML) pipeline for binary classification tasks. The pipeline includes feature selection, feature engineering, model training, and blending of predictions from multiple models. It leverages **AutoGluon** and **LightAutoML** for model training and optimization.

---

## 🚀 **What does this code do?**

The pipeline performs the following steps:

1. **Data Preprocessing**:
   - Removes non-numeric columns and handles missing values.
   - Filters out columns with low variance.

2. **Feature Selection**:
   - Uses LightGBM to select the top 15 most important features based on feature importance.

3. **Feature Engineering**:
   - Generates new features by combining existing ones (multiplication, division, trigonometric transformations).

4. **Feature Selection with Boruta**:
   - Applies the Boruta algorithm to further refine the feature set.

5. **Data Balancing**:
   - Downsamples the majority class to handle class imbalance.

6. **Model Training**:
   - Trains models using **AutoGluon** and **LightAutoML**.
   - Optimizes for ROC-AUC metric.

7. **Blending Predictions**:
   - Blends predictions from AutoGluon and LightAutoML using an optimal weight combination.

8. **Evaluation**:
   - Evaluates the blended model using ROC-AUC.

---

## 🛠️ **Requirements**

To run this code, you need the following dependencies:

- Python 3.8+
- Libraries listed in `requirements.txt`

### Install Dependencies

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/automl-pipeline.git
   cd automl-pipeline
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ **How to Run**

1. **Prepare Your Data**:
   - Ensure your data is in a `.parquet` file with a binary target column named `target`.

2. **Run the Pipeline**:
   - Execute the `main.py` script:
     ```bash
     python main.py
     ```

3. **Output**:
   - The script will output the best weights for blending and the corresponding ROC-AUC score.

---

## 🚨 **Important Notes**

### GPU Support
To leverage GPU acceleration (recommended for faster training), ensure you have the following:

1. **NVIDIA GPU**:
   - Verify that your system has an NVIDIA GPU.

2. **CUDA Toolkit**:
   - Install the appropriate version of CUDA Toolkit for your GPU. Download it from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

3. **NVIDIA Drivers**:
   - Install the latest NVIDIA drivers for your GPU.

4. **PyTorch with CUDA**:
   - Install PyTorch with CUDA support:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

5. **AutoGluon with GPU**:
   - Ensure AutoGluon is installed with GPU support:
     ```bash
     pip install "autogluon[all]"
     ```

---

## 📂 **File Structure**

```
automl-pipeline/
├── main.py                     # Main script to run the pipeline
├── requirements.txt            # List of dependencies
├── README.md                   # This file
└── data.parquet                # Example dataset (replace with your data)
```

---

## 📊 **Example Output**

When you run the pipeline, you will see output similar to the following:

```
ROC-AUC Selection Model Score: 0.9123
Новые признаки созданы
Производительность AG: {'roc_auc': 0.925}
Производительность LAMA: 0.9274
Наилучшие веса: AutoGluon = 0.45, LightAutoML = 0.55
Наилучший ROC AUC: 0.9301
```

---

## 🤝 **Contributing**

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

---

## 📜 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📧 **Contact**

For questions or feedback, feel free to reach out:

- **Email**: nikita26.08.98nesterov@gmail.com
- **GitHub**: [AKAYO](https://github.com/akayooo)

---

Happy modeling! 🎉
