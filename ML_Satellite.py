import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import cv2
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import random
import tensorflow as tf
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

sns.set_style("whitegrid")

DATA_DIR = 'data'
CLASSES = ['cloudy', 'desert', 'green_area', 'water']
IMG_SIZE = 96  
BATCH_SIZE = 32  

OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Data directory: {DATA_DIR}")
print(f"Classes: {CLASSES}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")

def augment_image(img):
    img_uint8 = (img * 255).astype(np.uint8)
    
    alpha = 1.2  
    img_contrast = cv2.convertScaleAbs(img_uint8, alpha=alpha, beta=0)
    
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    img_sharp = cv2.filter2D(img_contrast, -1, kernel)
    
    return img_sharp.astype(np.float32) / 255.0

def load_data():
    start_time = time.time()
    
    images = []
    labels = []
    
    max_images_per_class = 1000
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, class_name)
        print(f"Processing class {class_name}")
        
        files = os.listdir(class_dir)
        image_files = [f for f in files if f.endswith('.jpg')]
        
        if len(image_files) > max_images_per_class:
            np.random.shuffle(image_files)
            image_files = image_files[:max_images_per_class]
            
        print(f"Using {len(image_files)} images")
        
        for file in tqdm(image_files, desc=f"Loading {class_name} images"):
            img_path = os.path.join(class_dir, file)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                img_enhanced = augment_image(img / 255.0)
                
                images.append(img_enhanced)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
    
    X = np.array(images)
    y = np.array(labels)
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
    
    print(f"Data loading completed, time taken: {time.time() - start_time:.2f} seconds")
    print(f"Training set size: {X_train.shape}, Validation set size: {X_val.shape}, Test set size: {X_test.shape}")
    print(f"Training set class distribution: {np.bincount(y_train)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def explore_data(X_train, y_train):
    class_counts = pd.Series(y_train).value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Training Set Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(class_counts.index, [CLASSES[i] for i in class_counts.index])
    
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 5, str(v), ha='center')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(10, 8))
    for i in range(4):
        class_indices = np.where(y_train == i)[0]
        
        if len(class_indices) > 0:
            idx = np.random.choice(class_indices)
            
            plt.subplot(2, 2, i + 1)
            plt.imshow(X_train[idx])
            plt.title(f"{CLASSES[i]}", fontsize=14)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images.png'))
    plt.close()
    

def apply_pca(X_train, X_test, X_val, y_train, n_components=100):
    start_time = time.time()
    
    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]
    n_samples_val = X_val.shape[0]
    
    X_train_flat = X_train.reshape(n_samples_train, -1)
    X_test_flat = X_test.reshape(n_samples_test, -1)
    X_val_flat = X_val.reshape(n_samples_val, -1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"PCA dimensionality reduction completed, from {X_train_flat.shape[1]} dimensions to {n_components} dimensions")
    print(f"First {n_components} principal components explain {cumulative_variance[-1]*100:.2f}% of variance")
    print(f"PCA processing time: {time.time() - start_time:.2f} seconds")
    
    return X_train_pca, X_test_pca, X_val_pca, pca, scaler

def train_svm(X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test):
    start_time = time.time()
    
    svm_model = SVC(C=100, kernel='rbf', gamma=0.01, probability=True, random_state=42)
    
    svm_model.fit(X_train_pca, y_train)
    
    val_pred = svm_model.predict(X_val_pca)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"SVM accuracy on validation set: {val_accuracy:.4f}")
    
    y_pred = svm_model.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASSES)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"SVM model training completed, time taken: {time.time() - start_time:.2f} seconds")
    print(f"SVM accuracy on test set: {test_accuracy:.4f}")
    print("Classification report:")
    print(report)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('SVM Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'svm_confusion_matrix.png'))
    plt.close()
    
    return svm_model, test_accuracy

def create_advanced_cnn_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(CLASSES), activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_cnn(X_train, y_train, X_val, y_val, X_test, y_test):
    start_time = time.time()
    
    y_train_cat = to_categorical(y_train, num_classes=len(CLASSES))
    y_val_cat = to_categorical(y_val, num_classes=len(CLASSES))
    y_test_cat = to_categorical(y_test, num_classes=len(CLASSES))
    
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    model = create_advanced_cnn_model()
    
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
        ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, 'cnn_best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=50,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"CNN model training completed, time taken: {time.time() - start_time:.2f} seconds")
    print(f"Test accuracy: {test_acc:.4f}")
    
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    report = classification_report(y_test, y_pred, target_names=CLASSES)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("Classification report:")
    print(report)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_training_history.png'))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('CNN Confusion Matrix')
    plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_confusion_matrix.png'))
    plt.close()
    
    return model, test_acc, history

def compare_models(svm_accuracy, cnn_accuracy):
    plt.figure(figsize=(8, 5))
    models = ['SVM (with PCA)', 'Enhanced CNN']
    accuracies = [svm_accuracy, cnn_accuracy]
    
    ax = sns.barplot(x=models, y=accuracies)
    
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'))
    plt.close()
    
    print(f"SVM (with PCA) Accuracy: {svm_accuracy:.4f}")
    print(f"CNN Accuracy: {cnn_accuracy:.4f}")
    
    if svm_accuracy > cnn_accuracy:
        print("SVM model performs better")
    else:
        print("CNN model performs better")

def main():
    total_start_time = time.time()
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    explore_data(X_train, y_train)
    
    X_train_pca, X_test_pca, X_val_pca, pca, scaler = apply_pca(X_train, X_test, X_val, y_train, n_components=100)
    
    svm_model, svm_accuracy = train_svm(X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test)
    
    cnn_model, cnn_accuracy, history = train_cnn(X_train, y_train, X_val, y_val, X_test, y_test)
    
    compare_models(svm_accuracy, cnn_accuracy)
    
    total_time = time.time() - total_start_time
    print(f"\n Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()