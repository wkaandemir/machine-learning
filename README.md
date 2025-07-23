# Depression Prediction with Machine Learning

Bu proje, depresyon tahmini için çeşitli makine öğrenmesi modellerini karşılaştıran ve optimize eden bir çalışmadır. Gerçek Kaggle Mental Health verisi (290K+ kayıt) ile test edilmiş ve %94+ doğruluk oranı elde edilmiştir.

## 📁 Proje Yapısı

```
machine-learning/
├── src/
│   ├── models/
│   │   └── depression_classification.py     # Ana ML pipeline
│   ├── data/
│   │   ├── kaggle_depression_dataset.xlsx   # Kaggle Mental Health verisi (5K)
│   │   ├── depression_sample_dataset.xlsx   # Simulated test verisi (1K)
│   │   └── mental_health_dataset.csv        # Ham Kaggle verisi (290K+)
│   └── utils/
│       └── preprocessing.py                 # Veri ön işleme yardımcıları
├── outputs/
│   └── models/
│       └── randomized_search/
│           ├── model_comparison.png         # Model performans karşılaştırması
│           ├── shap_summary.png            # SHAP analiz grafiği
│           ├── shap_importance_bar.png     # SHAP önem sıralaması
│           └── model_results.txt           # Detaylı sonuçlar
├── requirements.txt                        # Python bağımlılıkları
├── CLAUDE.md                              # Claude Code talimatları
└── README.md                              # Proje dokümantasyonu
```

## 🚀 Özellikler

- **Çoklu Model Karşılaştırması**: Logistic Regression, Gradient Boosting, SVM
- **Hiperparametre Optimizasyonu**: RandomizedSearchCV ile
- **Veri Dengeleme**: SMOTEENN ile sınıf dengesizliği giderimi
- **SHAP Analizi**: Model interpretability için
- **Kapsamlı Metrikler**: Accuracy, ROC-AUC, F1, Precision, Recall, PPV, NPV, Brier Score

## 📊 Model Performansları (Kaggle Dataset Sonuçları)

| Model | Accuracy | ROC-AUC | F1-Score | Precision | Recall |
|-------|----------|---------|----------|-----------|--------|
| **Gradient Boosting** ⭐ | **94.04%** | **98.13%** | **93.15%** | **95.58%** | **90.88%** |
| SVM | 93.93% | 97.52% | 93.08% | 94.85% | 91.41% |
| Logistic Regression | 82.20% | 86.51% | 80.41% | 79.18% | 81.68% |

*5000 örneklem üzerinde 5-fold cross-validation ile test edilmiştir.*

## 🔧 Kurulum

### 1. Conda Ortamı Kurulumu (Önerilen)

```bash
# Miniconda'yı indirin ve kurun
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Conda'yı bash için aktifleştirin
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# Conda kullanım şartlarını kabul edin
source $HOME/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Python 3.11 ile sanal ortam oluşturun
conda create -n ml-env python=3.11 -y

# Ortamı aktifleştirin
conda activate ml-env

# Gerekli paketleri kurun
pip install scikit-learn>=1.3.0 pandas>=1.5.0 numpy>=1.24.0 matplotlib>=3.6.0
pip install shap>=0.42.0 imbalanced-learn>=0.10.0 xgboost>=1.7.0
pip install scipy>=1.10.0 openpyxl>=3.1.0 seaborn>=0.12.0 jupyter>=1.0.0
```

### 2. Alternatif: Pip ile Kurulum

```bash
git clone https://github.com/username/depression-prediction-ml.git
cd depression-prediction-ml
pip install -r requirements.txt
```

### 3. Ortam Doğrulaması

```bash
# Conda ortamını test edin
conda activate ml-env
python -c "import sklearn, pandas, numpy, matplotlib, shap; print('✅ Tüm paketler başarıyla kuruldu!')"
```

## 📈 Kullanım

### 1. Conda Ortamını Aktifleştirme
```bash
# Conda ortamını aktifleştirin
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ml-env
```

### 2. Örnek Dataset ile Test
Bu proje Kaggle'dan gerçek mental health verisi ile test edilmiştir:
```bash
# Projeyi klonlayın
git clone <repository-url>
cd machine-learning

# Kaggle Mental Health Dataset ile test edin (5K örneklem dahil)
python src/models/depression_classification.py
# Prompt'ta: src/data/kaggle_depression_dataset.xlsx
```

### 3. Kendi Verinizi Kullanma
```bash
# Veri dosyanızı src/data/ klasörüne koyun
cp your_data_file.xlsx src/data/my_depression_data.xlsx

# Analizi çalıştırın
python src/models/depression_classification.py
# Prompt'ta: src/data/my_depression_data.xlsx
```

### 4. Komut Satırı Parametreleri
```bash
# Tüm parametrelerle çalıştırma
python src/models/depression_classification.py \
  --data-path src/data/kaggle_depression_dataset.xlsx \
  --n-iter 20 \
  --cv-folds 10 \
  --output-dir outputs/custom_results
```

### 5. Sonuçları İnceleyin
Analiz tamamlandıktan sonra `outputs/models/randomized_search/` klasöründe:
- `model_comparison.png`: Model performans karşılaştırması
- `shap_summary.png`: SHAP özellik önem analizi
- `shap_importance_bar.png`: SHAP bar grafiği
- `model_results.txt`: Detaylı sonuçlar ve parametreler

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

Kaggle Mental Health Dataset'i üzerinde yapılan SHAP analizi sonuçları:

1. **Family_History** (3.99) - Ailede mental sağlık geçmişi
2. **Country** (2.04) - Yaşanılan ülke
3. **Age** (1.47) - Yaş
4. **Mood_Swings** (0.88) - Ruh hali dalgalanmaları
5. **Work_Interest** (0.65) - İş ilgisi
6. **Mental_Health_History** (0.59) - Kişisel mental sağlık geçmişi
7. **Days_Indoors** (0.51) - Evde geçirilen gün sayısı
8. **Coping_Struggles** (0.49) - Başa çıkma zorlukları
9. **Social_Weakness** (0.45) - Sosyal zayıflık
10. **Growing_Stress** (0.42) - Artan stres

## 📝 Lisans

MIT License

## 👥 Katkıda Bulunanlar

- [Your Name]

## 📞 İletişim

- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername) 