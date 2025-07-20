"""
Data Preprocessing Utilities

Bu modül, veri ön işleme işlemleri için yardımcı fonksiyonlar içerir.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN
from collections import Counter


def check_missing_data(data):
    """
    Eksik veri analizi yapar
    
    Args:
        data (pd.DataFrame): Analiz edilecek veri seti
        
    Returns:
        pd.Series: Eksik veri sayıları
    """
    missing_data = data.isnull().sum()
    print("Eksik Veri Analizi:")
    print(missing_data[missing_data > 0])
    return missing_data


def fill_missing_values(data, strategy='mean'):
    """
    Eksik verileri doldurur
    
    Args:
        data (pd.DataFrame): Veri seti
        strategy (str): Doldurma stratejisi ('mean', 'median', 'mode')
        
    Returns:
        pd.DataFrame: Eksik verileri doldurulmuş veri seti
    """
    data_filled = data.copy()
    
    for col in data.select_dtypes(include=["number"]).columns:
        if data[col].isnull().sum() > 0:
            if strategy == 'mean':
                data_filled[col] = data_filled[col].fillna(data_filled[col].mean())
            elif strategy == 'median':
                data_filled[col] = data_filled[col].fillna(data_filled[col].median())
            elif strategy == 'mode':
                data_filled[col] = data_filled[col].fillna(data_filled[col].mode()[0])
    
    print(f"Eksik veriler '{strategy}' stratejisi ile dolduruldu.")
    return data_filled


def analyze_class_imbalance(y):
    """
    Sınıf dengesizliği analizi yapar
    
    Args:
        y (pd.Series): Hedef değişken
        
    Returns:
        dict: Dengesizlik analizi sonuçları
    """
    class_counts = Counter(y)
    imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
    
    print("Veri Dengesizliği Analizi:")
    print(f"Sınıf dağılımı: {class_counts}")
    print(f"Dengesizlik oranı: {imbalance_ratio:.2f}")
    
    return {
        'class_counts': class_counts,
        'imbalance_ratio': imbalance_ratio
    }


def balance_dataset(X, y, method='smoteenn', random_state=42):
    """
    Veri setini dengeler
    
    Args:
        X (pd.DataFrame): Özellik matrisi
        y (pd.Series): Hedef değişken
        method (str): Dengeleme yöntemi ('smoteenn', 'smote', 'random_under')
        random_state (int): Rastgele durum
        
    Returns:
        tuple: Dengelenmiş X ve y
    """
    if method == 'smoteenn':
        balancer = SMOTEENN(random_state=random_state)
    else:
        raise ValueError(f"Desteklenmeyen dengeleme yöntemi: {method}")
    
    X_resampled, y_resampled = balancer.fit_resample(X, y)
    
    print(f"{method.upper()} ile veri dengeleme yapıldı.")
    print(f"Yeni sınıf dağılımı: {Counter(y_resampled)}")
    
    return X_resampled, y_resampled


def encode_categorical_features(X, encoding_method='label'):
    """
    Kategorik değişkenleri kodlar
    
    Args:
        X (pd.DataFrame): Özellik matrisi
        encoding_method (str): Kodlama yöntemi ('label', 'onehot')
        
    Returns:
        pd.DataFrame: Kodlanmış özellik matrisi
    """
    X_encoded = X.copy()
    categorical_cols = X.select_dtypes(include=["object"]).columns
    
    if encoding_method == 'label':
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
        print(f"{len(categorical_cols)} kategorik değişken label encoding ile dönüştürüldü.")
    
    elif encoding_method == 'onehot':
        X_encoded = pd.get_dummies(X_encoded, columns=categorical_cols)
        print(f"{len(categorical_cols)} kategorik değişken one-hot encoding ile dönüştürüldü.")
    
    else:
        raise ValueError(f"Desteklenmeyen kodlama yöntemi: {encoding_method}")
    
    return X_encoded


def get_feature_info(X):
    """
    Özellik bilgilerini döndürür
    
    Args:
        X (pd.DataFrame): Özellik matrisi
        
    Returns:
        dict: Özellik bilgileri
    """
    info = {
        'total_features': X.shape[1],
        'numerical_features': len(X.select_dtypes(include=["number"]).columns),
        'categorical_features': len(X.select_dtypes(include=["object"]).columns),
        'feature_names': list(X.columns),
        'numerical_columns': list(X.select_dtypes(include=["number"]).columns),
        'categorical_columns': list(X.select_dtypes(include=["object"]).columns)
    }
    
    print("Özellik Bilgileri:")
    print(f"Toplam özellik sayısı: {info['total_features']}")
    print(f"Sayısal özellik sayısı: {info['numerical_features']}")
    print(f"Kategorik özellik sayısı: {info['categorical_features']}")
    
    return info


def preprocess_data(data, target_column, missing_strategy='mean', 
                   balance_method='smoteenn', encoding_method='label'):
    """
    Tam veri ön işleme pipeline'ı
    
    Args:
        data (pd.DataFrame): Ham veri seti
        target_column (str): Hedef değişken adı
        missing_strategy (str): Eksik veri doldurma stratejisi
        balance_method (str): Veri dengeleme yöntemi
        encoding_method (str): Kategorik değişken kodlama yöntemi
        
    Returns:
        tuple: (X_processed, y_processed, preprocessing_info)
    """
    print("Veri ön işleme başlıyor...")
    
    # Hedef değişkeni ayır
    X = data.drop([target_column], axis=1)
    y = data[target_column]
    
    # Eksik veri analizi ve doldurma
    check_missing_data(data)
    data_filled = fill_missing_values(data, missing_strategy)
    X_filled = data_filled.drop([target_column], axis=1)
    
    # Sınıf dengesizliği analizi
    imbalance_info = analyze_class_imbalance(y)
    
    # Veri dengeleme
    X_balanced, y_balanced = balance_dataset(X_filled, y, balance_method)
    
    # Kategorik değişken kodlama
    X_encoded = encode_categorical_features(X_balanced, encoding_method)
    
    # Özellik bilgileri
    feature_info = get_feature_info(X_encoded)
    
    preprocessing_info = {
        'imbalance_info': imbalance_info,
        'feature_info': feature_info,
        'missing_strategy': missing_strategy,
        'balance_method': balance_method,
        'encoding_method': encoding_method
    }
    
    print("Veri ön işleme tamamlandı!")
    
    return X_encoded, y_balanced, preprocessing_info 