# Depression Prediction with Machine Learning

Bu proje, depresyon tahmini için çeşitli makine öğrenmesi modellerini karşılaştıran ve optimize eden bir çalışmadır.

## 📁 Proje Yapısı

```
depression-prediction-ml/
├── src/
│   ├── models/
│   │   └── depression_classification.py
│   ├── data/
│   │   └── depresyon.xlsx (veri dosyasını buraya koyun)
│   └── utils/
│       └── preprocessing.py
├── outputs/
│   ├── models/
│   │   └── randomized_search/
│   │       ├── model_comparison.png
│   │       ├── shap_summary.png
│   │       ├── shap_importance_bar.png
│   │       └── model_results.txt
│   └── reports/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
└── README.md
```

## 🚀 Özellikler

- **Çoklu Model Karşılaştırması**: Logistic Regression, Gradient Boosting, SVM
- **Hiperparametre Optimizasyonu**: RandomizedSearchCV ile
- **Veri Dengeleme**: SMOTEENN ile sınıf dengesizliği giderimi
- **SHAP Analizi**: Model interpretability için
- **Kapsamlı Metrikler**: Accuracy, ROC-AUC, F1, Precision, Recall, PPV, NPV, Brier Score

## 📊 Model Performansları

| Model | Accuracy | ROC-AUC | F1-Score | Precision | Recall |
|-------|----------|---------|----------|-----------|--------|
| Gradient Boosting | 0.9318 | 0.9717 | 0.9311 | 0.9535 | 0.9101 |
| SVM | 0.8961 | 0.9500 | 0.8970 | 0.8979 | 0.8974 |
| Logistic Regression | 0.8180 | 0.9014 | 0.8238 | 0.8033 | 0.8464 |

## 🔧 Kurulum

```bash
git clone https://github.com/username/depression-prediction-ml.git
cd depression-prediction-ml
pip install -r requirements.txt
```

## 📈 Kullanım

### 1. Veri Dosyasını Yerleştirme
Veri dosyanızı `src/data/` klasörüne koyun:
```bash
# Veri dosyanızı src/data/ klasörüne kopyalayın
cp your_data_file.xlsx src/data/depresyon.xlsx
```

### 2. Analizi Çalıştırma
```python
# Ana model dosyasını çalıştırın
python src/models/depression_classification.py
```

Program çalıştığında veri dosyasının yolunu soracaktır. `src/data/depresyon.xlsx` yolunu girebilirsiniz.

### 3. Keşifsel Veri Analizi
```bash
# Jupyter notebook'u çalıştırın
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## 📋 Gereksinimler

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- shap
- imbalanced-learn
- xgboost

## 📝 Veri Formatı

Veri dosyanız aşağıdaki formatta olmalıdır:
- Excel (.xlsx) formatında
- Hedef değişken sütunu: "Depression"
- Diğer sütunlar: Özellik değişkenleri

## 🔍 Önemli Değişkenler (SHAP Analizi Sonuçları)

En önemli özellikler (önem sırasına göre):
1. Child_Age (8.50)
2. Having_shelter_problems_now (4.95)
3. Relationship_with_the_father_now (3.35)
4. Getting_support_in_an_earthquake (2.33)
5. Having_shelter_problems_in_an_earthquake (2.33)

## 📝 Lisans

MIT License

## 👥 Katkıda Bulunanlar

- [Your Name]

## 📞 İletişim

- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername) 