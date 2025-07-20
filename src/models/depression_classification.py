"""
Depression Classification Model with Randomized Search CV

Bu modül, depresyon tahmini için çeşitli makine öğrenmesi modellerini
karşılaştıran ve optimize eden bir sınıflandırma sistemi içerir.

Özellikler:
- Çoklu model karşılaştırması (Logistic Regression, Gradient Boosting, SVM)
- RandomizedSearchCV ile hiperparametre optimizasyonu
- SMOTEENN ile veri dengeleme
- SHAP analizi ile model interpretability
- Kapsamlı performans metrikleri
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, recall_score
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import shap
from imblearn.combine import SMOTEENN
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import os
import sys

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class DepressionClassifier:
    """
    Depresyon sınıflandırma modeli sınıfı
    """
    
    def __init__(self, data_path=None):
        """
        Args:
            data_path (str): Veri setinin yolu (None ise kullanıcıdan istenir)
        """
        if data_path is None:
            print("Lütfen veri dosyasının yolunu belirtin:")
            print("Örnek: 'data/depresyon.xlsx' veya tam yol")
            self.data_path = input("Veri dosyası yolu: ")
        else:
            self.data_path = data_path
            
        self.data = None
        self.X = None
        self.y = None
        self.X_resampled = None
        self.y_resampled = None
        self.best_models = {}
        self.best_params = {}
        self.all_metrics = {}
        
    def load_data(self):
        """Veri setini yükler ve temel analizleri yapar"""
        print("Veri seti yükleniyor...")
        try:
            self.data = pd.read_excel(self.data_path)
            self.X = self.data.drop(["Depression"], axis=1)
            self.y = self.data["Depression"]
            
            print(f"Veri seti boyutu: {self.data.shape}")
            print(f"Özellik sayısı: {self.X.shape[1]}")
            print(f"Hedef değişken dağılımı: {Counter(self.y)}")
        except FileNotFoundError:
            print(f"Hata: {self.data_path} dosyası bulunamadı!")
            print("Lütfen doğru dosya yolunu belirtin.")
            return False
        except Exception as e:
            print(f"Hata: {e}")
            return False
        return True
        
    def handle_missing_data(self):
        """Eksik verileri kontrol eder ve doldurur"""
        print("\nEksik veri analizi yapılıyor...")
        missing_data = self.data.isnull().sum()
        print("Eksik veri sayıları:")
        print(missing_data[missing_data > 0])
        
        # Sayısal sütunlardaki eksik verileri ortalama ile doldur
        for col in self.data.select_dtypes(include=["number"]).columns:
            self.data[col] = self.data[col].fillna(self.data[col].mean())
            
        print("Eksik veriler dolduruldu.")
        
    def balance_data(self):
        """Veri dengesizliğini kontrol eder ve SMOTEENN ile dengeler"""
        print("\nVeri dengesizliği analizi yapılıyor...")
        class_counts = Counter(self.y)
        imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
        print(f"Dengesizlik oranı: {imbalance_ratio:.2f}")
        
        # SMOTEENN ile dengeleme
        print("SMOTEENN ile veri dengeleme yapılıyor...")
        smote_enn = SMOTEENN(random_state=42)
        self.X_resampled, self.y_resampled = smote_enn.fit_resample(self.X, self.y)
        print(f"SMOTEENN sonrası sınıf dağılımı: {Counter(self.y_resampled)}")
        
    def encode_categorical_features(self):
        """Kategorik değişkenleri sayısal değerlere dönüştürür"""
        print("\nKategorik değişkenler dönüştürülüyor...")
        categorical_cols = self.X_resampled.select_dtypes(include=["object"]).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.X_resampled[col] = le.fit_transform(self.X_resampled[col])
            
        print("Kategorik değişkenler dönüştürüldü.")
        
    def optimize_and_evaluate_model(self, model, param_space, X, y, model_name):
        """
        Modeli optimize eder ve değerlendirir
        
        Args:
            model: Sklearn model nesnesi
            param_space (dict): Hiperparametre uzayı
            X: Özellik matrisi
            y: Hedef değişken
            model_name (str): Model adı
            
        Returns:
            tuple: (en iyi model, en iyi parametreler, metrikler)
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # RandomizedSearchCV kullanımı
        opt = RandomizedSearchCV(
            model,
            param_space,
            n_iter=2,  # Rastgele arama sayısı
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        opt.fit(X, y)
        print(f"\n{model_name} - En iyi parametreler:", opt.best_params_)
        
        # En iyi model ile değerlendirme
        best_model = model.__class__(**opt.best_params_, random_state=42 if hasattr(model, 'random_state') else None)
        
        # Cross-validation metrikleri
        cv_results = {
            'accuracy': [], 'roc_auc': [], 'brier_score': [],
            'recall': [], 'precision': [], 'f1': [],
            'ppv': [], 'npv': []
        }
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            cv_results['accuracy'].append(accuracy_score(y_test, y_pred))
            cv_results['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))
            cv_results['brier_score'].append(brier_score_loss(y_test, y_pred_proba))
            cv_results['recall'].append(recall_score(y_test, y_pred))
            cv_results['precision'].append(precision_score(y_test, y_pred))
            cv_results['f1'].append(f1_score(y_test, y_pred))
            cv_results['ppv'].append(ppv)
            cv_results['npv'].append(npv)
        
        metrics = {metric: np.mean(values) for metric, values in cv_results.items()}
        
        print(f"\n{model_name} 5-Fold Cross Validation Sonuçları:")
        for metric_name, value in metrics.items():
            print(f"{metric_name.title()}: {value:.4f} (±{np.std(cv_results[metric_name]):.4f})")
        
        return best_model, opt.best_params_, metrics
        
    def define_models_and_params(self):
        """Model ve parametre uzaylarını tanımlar"""
        return {
            "Logistic Regression": {
                "model": LogisticRegression(random_state=42, max_iter=2000),
                "params": {
                    "C": uniform(0.1, 100.0),
                    "solver": ['lbfgs'],
                    "penalty": ['l2'],
                    "class_weight": ['balanced', None]
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": randint(100, 501),
                    "learning_rate": uniform(0.01, 0.19),
                    "max_depth": randint(3, 9),
                    "min_samples_split": randint(2, 11),
                    "min_samples_leaf": randint(1, 5),
                    "subsample": uniform(0.8, 0.2)
                }
            },
            "SVM": {
                "model": SVC(random_state=42),
                "params": {
                    "C": uniform(0.1, 100),
                    "kernel": ['rbf', 'linear'],
                    "gamma": ['scale', 'auto', 0.001, 0.01, 0.1],
                    "class_weight": ['balanced', None],
                    "probability": [True]
                }
            }
        }
        
    def train_and_evaluate_models(self):
        """Tüm modelleri eğitir ve değerlendirir"""
        models_and_params = self.define_models_and_params()
        
        print("\nModel optimizasyonu ve değerlendirmesi başlıyor...")
        print("=" * 50)
        
        for model_name, model_info in models_and_params.items():
            print(f"\n{model_name} modeli optimize ediliyor...")
            best_model, best_param, metrics = self.optimize_and_evaluate_model(
                model_info["model"],
                model_info["params"],
                self.X_resampled, self.y_resampled,
                model_name
            )
            self.best_models[model_name] = best_model
            self.best_params[model_name] = best_param
            self.all_metrics[model_name] = metrics
            print("=" * 50)
            
    def create_visualizations(self, output_dir="../../outputs/models/randomized_search"):
        """Sonuçları görselleştirir"""
        print("\nGörselleştirmeler oluşturuluyor...")
        
        # Çıktı klasörünü oluştur
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Model karşılaştırma grafiği
        plt.figure(figsize=(15, 10))
        metrics_to_plot = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall', 'ppv', 'npv', 'brier_score']
        x = np.arange(len(self.all_metrics))
        width = 0.11
        multiplier = 0
        
        for metric in metrics_to_plot:
            offset = width * multiplier
            values = [self.all_metrics[model][metric] for model in self.all_metrics.keys()]
            plt.bar(x + offset, values, width, label=metric.replace('_', ' ').title())
            multiplier += 1
        
        plt.xlabel('Models')
        plt.ylabel('Scores')
        plt.title('Model Performance Comparison (Randomized Search)')
        plt.xticks(x + width * 3.5, self.all_metrics.keys(), rotation=45)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), bbox_inches='tight')
        plt.close()
        
        print("Model karşılaştırma grafiği kaydedildi.")
        
    def save_results(self, output_dir="../../outputs/models/randomized_search"):
        """Sonuçları dosyaya kaydeder"""
        print("\nSonuçlar dosyaya kaydediliyor...")
        
        # En iyi modeli bul
        best_model_name = max(self.all_metrics.items(), key=lambda x: x[1]['accuracy'])[0]
        
        # Sonuçları yazdır
        print(f"\nEn iyi performans gösteren model: {best_model_name}")
        print("\nTüm modellerin performans özeti Randomized Search ile:")
        print("=" * 110)
        print(f"{'Model':<20} {'Accuracy':<10} {'ROC-AUC':<10} {'F1':<10} {'PPV':<10} {'NPV':<10} {'Brier':<10} {'Precision':<10} {'Recall':<10}")
        print("=" * 110)
        
        for model_name, metrics in self.all_metrics.items():
            print(f"{model_name:<20} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['roc_auc']:<10.4f} "
                  f"{metrics['f1']:<10.4f} "
                  f"{metrics['ppv']:<10.4f} "
                  f"{metrics['npv']:<10.4f} "
                  f"{metrics['brier_score']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f}")
        
        # Sonuçları dosyaya kaydet
        with open(os.path.join(output_dir, 'model_results.txt'), 'w', encoding='utf-8') as f:
            f.write("RANDOMIZED SEARCH SONUÇLARI\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"EN İYİ MODEL: {best_model_name}\n")
            f.write(f"Accuracy: {self.all_metrics[best_model_name]['accuracy']:.4f}\n")
            f.write("-" * 50 + "\n\n")
            
            f.write("TÜM MODELLERİN SONUÇLARI:\n")
            for model_name, metrics in self.all_metrics.items():
                f.write(f"\n{model_name}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"- {metric_name}: {value:.4f}\n")
                f.write("-" * 30 + "\n")
            
            f.write("\nEN İYİ PARAMETRELER:\n")
            for model_name, params in self.best_params.items():
                f.write(f"\n{model_name}:\n")
                for param, value in params.items():
                    f.write(f"- {param}: {value}\n")
                    
        print("Sonuçlar dosyaya kaydedildi.")
        
    def perform_shap_analysis(self, output_dir="../../outputs/models/randomized_search"):
        """SHAP analizi yapar"""
        print("\nSHAP analizi yapılıyor...")
        
        # En iyi modeli bul
        best_model_name = max(self.all_metrics.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model = self.best_models[best_model_name]
        
        if isinstance(best_model, (GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(self.X_resampled)
            
            # SHAP görselleştirmeleri
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, self.X_resampled, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_summary.png'), bbox_inches='tight', dpi=300)
            plt.close()
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, self.X_resampled, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_importance_bar.png'), bbox_inches='tight', dpi=300)
            plt.close()
            
            # Değişken önem değerlerini txt dosyasına yazma
            with open(os.path.join(output_dir, 'model_results.txt'), 'a', encoding='utf-8') as f:
                f.write("\nDEĞİŞKEN ÖNEM DEĞERLERİ (SHAP):\n")
                f.write("=" * 50 + "\n")
                
                # SHAP değerlerinin mutlak ortalamasını hesaplama
                feature_importance = np.abs(shap_values).mean(0)
                feature_importance_dict = dict(zip(self.X_resampled.columns, feature_importance))
                
                # Önem değerlerine göre sıralama
                sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                # Değerleri yazdırma
                for feature, importance in sorted_features:
                    f.write(f"{feature}: {importance:.4f}\n")
                    
            print("SHAP analizi tamamlandı.")
        else:
            print(f"\nSHAP analizi {best_model_name} modeli için desteklenmiyor.")
            
    def run_complete_analysis(self):
        """Tüm analiz sürecini çalıştırır"""
        print("Depresyon Sınıflandırma Analizi Başlıyor...")
        print("=" * 50)
        
        # Veri yükleme ve ön işleme
        if not self.load_data():
            return False
            
        self.handle_missing_data()
        self.balance_data()
        self.encode_categorical_features()
        
        # Model eğitimi ve değerlendirme
        self.train_and_evaluate_models()
        
        # Sonuçları kaydetme ve görselleştirme
        self.create_visualizations()
        self.save_results()
        self.perform_shap_analysis()
        
        print("\nAnaliz tamamlandı!")
        print("Sonuçlar 'outputs/models/randomized_search' klasörüne kaydedildi.")
        return True


def main():
    """Ana fonksiyon"""
    classifier = DepressionClassifier()
    classifier.run_complete_analysis()


if __name__ == "__main__":
    main() 