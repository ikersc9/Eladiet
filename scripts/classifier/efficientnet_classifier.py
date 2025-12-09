"""
EfficientNet-B0 Fine-tuned Classifier para Pastillas OK/NOK
VERSIÓN HÍBRIDA: Combina soluciones de desbalance + Recentrado + Máscara Dinámica
- Robustez: Estrategia jerárquica (Otsu + Canny fallback)
- Máscara: Gaussiana adaptativa basada en contorno elíptico
- Desbalance: Class weights + Data augmentation inteligente
- Métricas completas: Accuracy, Precision, Recall, AUC, TP, FP, TN, FN
"""

import numpy as np
import cv2
import os
from pathlib import Path
import pickle
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class EfficientNetPillClassifier:
    def __init__(self, img_size=(300, 300), apply_dynamic_mask=True):
        """
        Clasificador EfficientNet con preprocesamiento geométrico avanzado
        
        Args:
            img_size: Tamaño de entrada (300x300 para máxima resolución de detalles)
            apply_dynamic_mask: Activar máscara gaussiana adaptativa
        """
        self.img_size = img_size
        self.apply_dynamic_mask = apply_dynamic_mask
        self.model = None
        self.history = None
        self.trained = False
        self.class_names = ['OK', 'NOK']
        self.optimal_threshold = 0.5
        
    def _center_pill_in_image(self, img):
        """
        ESTRATEGIA JERÁRQUICA DE RECENTRADO:
        1. Otsu threshold (contraste claro)
        2. Canny edges (fallback para bajo contraste)
        3. Transformación geométrica para centrar el centroide
        
        Returns:
            img_centered: Imagen con pastilla centrada
            main_contour_shifted: Contorno desplazado (para máscara)
        """
        try:
            h, w = self.img_size
            img_u8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
            gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
            
            # FASE 1: Intentar con Umbral de Otsu (óptimo para buen contraste)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            main_contour = None
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                # Filtrar contornos demasiado pequeños (ruido)
                if cv2.contourArea(main_contour) < 500: 
                    main_contour = None
            
            # FASE 2: FALLBACK con Canny (sensible a gradientes locales)
            if main_contour is None:
                edges = cv2.Canny(gray, 50, 150)
                contours_canny, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours_canny:
                    main_contour = max(contours_canny, key=cv2.contourArea)
                    if cv2.contourArea(main_contour) < 50:
                        main_contour = None
            
            # Si no se detecta nada, devolver imagen sin modificar
            if main_contour is None:
                return img, None
            
            # FASE 3: Calcular centroide del contorno
            M = cv2.moments(main_contour)
            if M["m00"] == 0:
                return img, None
            
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # FASE 4: Transformación geométrica para centrar
            shift_x = (w // 2) - cX
            shift_y = (h // 2) - cY
            
            M_trans = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            
            img_centered = cv2.warpAffine(
                img, M_trans, (w, h), 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=(0, 0, 0)
            )
            
            # Desplazar el contorno para que coincida con la imagen centrada
            main_contour_shifted = cv2.transform(main_contour, M_trans)
            
            return img_centered, main_contour_shifted
            
        except Exception as e:
            print(f"⚠️ Warning en recentrado: {e}")
            return img, None
    
    def _apply_dynamic_mask(self, img, contour, strength=0.5):
        """
        MÁSCARA GAUSSIANA ADAPTATIVA:
        - Se ajusta a la forma real del contorno (elipse fitteada)
        - Suavizado gaussiano para transición gradual
        - Atenúa píxeles periféricos sin eliminarlos bruscamente
        
        Args:
            img: Imagen de entrada
            contour: Contorno detectado de la pastilla
            strength: Intensidad del enmascaramiento (0=sin efecto, 1=máximo)
        
        Returns:
            Imagen con máscara aplicada
        """
        try:
            h, w = img.shape[:2]
            
            # Si no hay contorno válido, devolver imagen sin modificar
            if contour is None or len(contour) < 5:
                return img
            
            # Ajustar elipse al contorno (se adapta a formas ovaladas)
            ellipse = cv2.fitEllipse(contour)
            
            # Crear máscara base (elipse rellena)
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(temp_mask, ellipse, 255, -1)
            
            # Aplicar desenfoque gaussiano para suavizar bordes
            mask = cv2.GaussianBlur(temp_mask, (71, 71), 0)
            
            # Normalizar [0, 1] con protección contra división por cero
            max_val = np.max(mask)
            mask = mask / (max_val + 1e-7)
            
            # Aplicar strength: 1 = sin atenuar, (1-strength) = máxima atenuación
            mask = 1 - strength * (1 - mask)
            
            # Convertir a 3 canales para aplicar sobre RGB
            mask_3ch = np.stack([mask] * 3, axis=-1)
            
            # Aplicar máscara
            return (img.astype(np.float32) * mask_3ch).astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Error en máscara dinámica: {e}")
            return img
    
    def load_and_preprocess_image(self, image_path):
        """
        PIPELINE COMPLETO DE PREPROCESAMIENTO:
        1. Carga y conversión RGB
        2. Redimensionamiento a img_size
        3. Recentrado geométrico (estrategia jerárquica)
        4. Máscara gaussiana adaptativa (opcional)
        5. Limpieza de NaN/Inf y clipping [0, 255]
        
        CRÍTICO: NO aplicar preprocess_input aquí (se hace en el modelo)
        """
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path
        
        if img is None or img.size == 0:
            return None
        
        # 1. Convertir BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Redimensionar
        img_resized = cv2.resize(img_rgb, self.img_size)
        
        # 3. RECENTRADO ROBUSTO
        img_centered, main_contour = self._center_pill_in_image(img_resized)
        
        # 4. MÁSCARA DINÁMICA (si está activada)
        img_final = img_centered.astype(np.float32)
        if self.apply_dynamic_mask:
            img_final = self._apply_dynamic_mask(img_final, main_contour, strength=0.5)
        
        # 5. LIMPIEZA CRÍTICA (previene loss=nan)
        img_final = np.nan_to_num(img_final, nan=0.0, posinf=255.0, neginf=0.0)
        img_final = np.clip(img_final, 0, 255)
        
        return img_final
    
    def build_model(self, learning_rate=1e-4, dropout=0.5):
        """
        Construye modelo EfficientNet-B0 con:
        - Base pre-entrenada en ImageNet
        - Cabezal personalizado con BatchNorm y Dropout
        - Métricas completas (Precision, Recall, AUC, TP/FP/TN/FN)
        """
        # Base model pre-entrenado
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Inicialmente congelado (Phase 1: entrenar solo cabezal)
        base_model.trainable = False
        
        # Construir arquitectura completa
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Preprocesamiento DENTRO del modelo (consistencia)
        x = preprocess_input(inputs)
        
        # Feature extraction
        x = base_model(x, training=False)
        
        # Cabezal clasificador
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(128, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        
        # Output binario
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compilar con métricas completas
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.TruePositives(name='tp'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.TrueNegatives(name='tn'),
                keras.metrics.FalseNegatives(name='fn')
            ]
        )
        
        print(f"\n✅ Modelo construido (LR={learning_rate})")
        print(f"   Total params: {self.model.count_params():,}")
        return self.model
    
    def unfreeze_model(self, learning_rate=1e-5):
        """
        Phase 2: Descongela las últimas 30 capas de EfficientNet para fine-tuning
        """
        # Encontrar la capa base
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, keras.Model) or 'efficientnet' in layer.name:
                base_model = layer
                break
        
        if base_model:
            base_model.trainable = True
            # Congelar todas excepto las últimas 30
            for layer in base_model.layers[:-30]:
                layer.trainable = False
            
            print(f"✅ Descongeladas últimas 30 capas de EfficientNet")
        
        # Re-compilar con LR reducido
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.TruePositives(name='tp'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.TrueNegatives(name='tn'),
                keras.metrics.FalseNegatives(name='fn')
            ]
        )
        print(f"✅ Modelo recompilado (LR={learning_rate})")
    
    def load_dataset(self, ok_folder, nok_folder, max_images_per_class=None):
        """
        Carga dataset con preprocesamiento completo
        """
        print(f"\n📂 CARGANDO DATASET")
        print(f"   Recentrado: ✅ Activo")
        print(f"   Máscara dinámica: {'✅ Activa' if self.apply_dynamic_mask else '❌ Desactivada'}")
        
        X, y = [], []
        
        # Cargar OK (label=0)
        ok_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            ok_paths.extend(Path(ok_folder).glob(ext))
        ok_paths = sorted(ok_paths)[:max_images_per_class] if max_images_per_class else sorted(ok_paths)
        
        for img_path in tqdm(ok_paths, desc="Cargando OK"):
            img = self.load_and_preprocess_image(str(img_path))
            if img is not None:
                X.append(img)
                y.append(0)
        
        n_ok = len([label for label in y if label == 0])
        print(f"✅ Cargadas {n_ok} imágenes OK")
        
        # Cargar NOK (label=1)
        nok_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            nok_paths.extend(Path(nok_folder).glob(ext))
        nok_paths = sorted(nok_paths)[:max_images_per_class] if max_images_per_class else sorted(nok_paths)
        
        for img_path in tqdm(nok_paths, desc="Cargando NOK"):
            img = self.load_and_preprocess_image(str(img_path))
            if img is not None:
                X.append(img)
                y.append(1)
        
        n_nok = len([label for label in y if label == 1])
        print(f"❌ Cargadas {n_nok} imágenes NOK")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        print(f"\n📊 Dataset final:")
        print(f"   Total: {len(X)} imágenes")
        print(f"   OK: {n_ok} ({n_ok/len(X)*100:.1f}%)")
        print(f"   NOK: {n_nok} ({n_nok/len(X)*100:.1f}%)")
        print(f"   Ratio desbalance: {n_ok/n_nok:.1f}:1")
        print(f"   Shape: {X.shape}")
        
        return X, y
    
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """Encuentra umbral que maximiza F1-score"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        f1_scores = []
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx], f1_scores[optimal_idx]
    
    def train(self, ok_folder, nok_folder, 
              epochs_phase1=10, 
              epochs_phase2=20, 
              batch_size=32,
              use_class_weights=True,
              augment_minority=True):
        """
        ENTRENAMIENTO EN 2 FASES:
        Phase 1: Solo cabezal (LR=1e-4)
        Phase 2: Fine-tuning últimas capas (LR=1e-5)
        
        Args:
            augment_minority: Augmentar clase NOK si ratio > 2:1
        """
        print("\n" + "="*80)
        print("🎯 ENTRENAMIENTO EFFICIENTNET + RECENTRADO + MÁSCARA DINÁMICA")
        print("="*80)
        
        # 1. Cargar datos
        X, y = self.load_dataset(ok_folder, nok_folder)
        
        # 2. Shuffle
        print(f"\n🔀 Mezclando datos...")
        X, y = shuffle(X, y, random_state=42)
        
        # 3. Split train/val/test (60/20/20)
        print(f"\n📊 Dividiendo dataset...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        print(f"   Train: {len(X_train)} imágenes")
        print(f"   Val: {len(X_val)} imágenes")
        print(f"   Test: {len(X_test)} imágenes")
        
        # 4. DATA AUGMENTATION (solo en train, solo si muy desbalanceado)
        n_ok_train = np.sum(y_train == 0)
        n_nok_train = np.sum(y_train == 1)
        
        if augment_minority and (n_ok_train / n_nok_train > 2.0):
            print(f"\n🎨 Data Augmentation (ratio {n_ok_train/n_nok_train:.1f}:1)...")
            
            # Augmentation pipeline
            data_augment = keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(1),
                layers.RandomBrightness(0.2),
                layers.RandomContrast(0.2),
            ])
            
            # Objetivo: llevar NOK al 40% de OK
            target_nok = int(n_ok_train * 0.4)
            n_augment = max(0, target_nok - n_nok_train)
            
            print(f"   Generando {n_augment} NOK adicionales...")
            
            nok_indices = np.where(y_train == 1)[0]
            X_train_list = list(X_train)
            y_train_list = list(y_train)
            
            for i in tqdm(range(n_augment), desc="   Augmentando"):
                idx = np.random.choice(nok_indices)
                img = X_train[idx]
                img_aug = data_augment(np.expand_dims(img, 0), training=True)[0].numpy()
                X_train_list.append(img_aug)
                y_train_list.append(1)
            
            X_train = np.array(X_train_list, dtype=np.float32)
            y_train = np.array(y_train_list, dtype=np.int32)
            X_train, y_train = shuffle(X_train, y_train, random_state=42)
            
            print(f"   ✅ Nuevo ratio: {n_ok_train}:{n_nok_train + n_augment} "
                  f"({n_ok_train/(n_nok_train + n_augment):.1f}:1)")
        
        # 5. CLASS WEIGHTS
        class_weights = None
        if use_class_weights:
            print(f"\n⚖️  Calculando class weights...")
            cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights = {0: cw[0], 1: cw[1]}
            print(f"   OK: {cw[0]:.2f}, NOK: {cw[1]:.2f}")
        
        # 6. Construir modelo
        if self.model is None:
            self.build_model(learning_rate=1e-4, dropout=0.5)
        
        # 7. Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=15, restore_best_weights=True, mode='max', verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
            )
        ]
        
        # 8. PHASE 1: Entrenar cabezal
        print(f"\n🚀 PHASE 1: Entrenando cabezal ({epochs_phase1} épocas, LR=1e-4)")
        history1 = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs_phase1,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # 9. PHASE 2: Fine-tuning
        print(f"\n🔥 PHASE 2: Fine-tuning ({epochs_phase2} épocas, LR=1e-5)")
        self.unfreeze_model(learning_rate=1e-5)
        history2 = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs_phase2,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combinar historiales
        self.history = keras.callbacks.History()
        self.history.history = {}
        for key in history1.history.keys():
            self.history.history[key] = history1.history[key] + history2.history[key]
        
        # 10. Calcular umbral óptimo en VAL
        print(f"\n🎯 Calculando umbral óptimo en validación...")
        y_val_pred_proba = self.model.predict(X_val, verbose=0).flatten()
        self.optimal_threshold, best_f1 = self.find_optimal_threshold(y_val, y_val_pred_proba)
        print(f"   Umbral óptimo: {self.optimal_threshold:.4f} (F1: {best_f1:.4f})")
        
        # 11. EVALUACIÓN FINAL EN TEST
        print("\n" + "="*80)
        print("📊 RESULTADOS FINALES EN TEST SET")
        print("="*80)
        
        y_test_proba = self.model.predict(X_test, verbose=0).flatten()
        y_test_pred = (y_test_proba >= self.optimal_threshold).astype(int)
        
        cm = confusion_matrix(y_test, y_test_pred)
        print("\nMatriz de Confusión:")
        print(f"              Pred OK  Pred NOK")
        print(f"Real OK      {cm[0,0]:6d}   {cm[0,1]:6d}")
        print(f"Real NOK     {cm[1,0]:6d}   {cm[1,1]:6d}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred, target_names=['OK', 'NOK']))
        
        auc = roc_auc_score(y_test, y_test_proba)
        print(f"🎯 AUC-ROC: {auc:.4f}")
        
        self.trained = True
        print("\n✅ Entrenamiento completado!")
        
        return self.history
    
    def predict_single(self, image_path, use_optimal_threshold=True):
        """Predicción individual con umbral óptimo"""
        if not self.trained:
            raise ValueError("Modelo no entrenado")
        
        img = self.load_and_preprocess_image(image_path)
        if img is None:
            return {'error': 'No se pudo cargar la imagen'}
        
        proba = self.model.predict(np.expand_dims(img, 0), verbose=0)[0][0]
        threshold = self.optimal_threshold if use_optimal_threshold else 0.5
        is_nok = proba >= threshold
        
        return {
            'classification': 'NOK' if is_nok else 'OK',
            'proba_nok': float(proba),
            'proba_ok': float(1 - proba),
            'is_nok': bool(is_nok),
            'threshold_used': float(threshold),
            'raw_score': float(proba)
        }
    
    def visualize_defect_gradcam(self, image_path, save_path=None, alpha=0.5):
        """
        Genera visualización Grad-CAM mostrando dónde el modelo detecta defectos
        
        Args:
            image_path: Ruta a la imagen o array
            save_path: Ruta donde guardar la visualización (opcional)
            alpha: Transparencia del heatmap (0-1)
        
        Returns:
            dict con resultado y visualización
        """
        if not self.trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Cargar imagen ORIGINAL (sin preprocesar) para visualización
        if isinstance(image_path, str):
            img_display = cv2.imread(image_path)
        else:
            img_display = image_path
        
        if img_display is None:
            return {'error': 'No se pudo cargar la imagen'}
        
        img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        img_display_resized = cv2.resize(img_display_rgb, self.img_size)
        
        # Cargar imagen PREPROCESADA para predicción
        img_preprocessed = self.load_and_preprocess_image(image_path)
        if img_preprocessed is None:
            return {'error': 'No se pudo cargar la imagen'}
        
        # Predecir con imagen preprocesada
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        proba = self.model.predict(img_batch, verbose=0)[0][0]
        
        # Clasificación
        threshold = self.optimal_threshold
        is_nok = proba >= threshold
        classification = 'NOK' if is_nok else 'OK'
        
        try:
            # Identificar la capa EfficientNet
            target_layer_name = None
            for layer in self.model.layers:
                if 'efficientnet' in layer.name.lower() or isinstance(layer, keras.Model):
                    target_layer_name = layer.name
                    break
            
            if target_layer_name is None:
                return {
                    'classification': classification,
                    'proba_nok': float(proba),
                    'error': 'No se encontró la capa EfficientNet'
                }
            
            # FORWARD PASS MANUAL para capturar activaciones intermedias
            with tf.GradientTape() as tape:
                x = tf.cast(img_batch, tf.float32)
                
                conv_outputs = None
                predictions = None
                
                # Iterar por todas las capas del modelo
                for layer in self.model.layers:
                    # Saltar InputLayer
                    if isinstance(layer, keras.layers.InputLayer):
                        continue
                    
                    # Aplicar capa
                    try:
                        x = layer(x, training=False)
                    except TypeError:
                        x = layer(x)
                    
                    # Capturar output de EfficientNet
                    if layer.name == target_layer_name and conv_outputs is None:
                        conv_outputs = x
                        tape.watch(conv_outputs)
                
                predictions = x
            
            if conv_outputs is None:
                return {
                    'classification': classification,
                    'proba_nok': float(proba),
                    'error': 'No se pudieron capturar las activaciones de EfficientNet'
                }
            
            # Calcular gradientes
            grads = tape.gradient(predictions, conv_outputs)
            
            # Global Average Pooling de los gradientes
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Ponderar las activaciones por los gradientes
            conv_outputs_np = conv_outputs[0].numpy()
            pooled_grads_np = pooled_grads.numpy()
            
            heatmap = conv_outputs_np @ pooled_grads_np[..., np.newaxis]
            heatmap = np.squeeze(heatmap)
            
            # Normalizar heatmap [0, 1]
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            # Redimensionar heatmap al tamaño de la imagen
            heatmap_resized = cv2.resize(heatmap, (self.img_size[0], self.img_size[1]))
            
            # Aplicar colormap (JET: azul=frío, rojo=caliente)
            heatmap_colored = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Superponer sobre imagen ORIGINAL (sin preprocesar)
            img_uint8 = img_display_resized.astype(np.uint8)
            superimposed = heatmap_colored * alpha + img_uint8 * (1 - alpha)
            superimposed = np.uint8(superimposed)
            
            # Guardar visualización si se especifica
            if save_path:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                # Original (sin preprocesar)
                axes[0].imshow(img_uint8)
                axes[0].set_title('Original', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                
                # Superposición (Detección)
                axes[1].imshow(superimposed)
                color = 'red' if is_nok else 'green'
                axes[1].set_title(f'Detección: {classification}\nP(NOK): {proba:.1%}', 
                                 fontsize=14, fontweight='bold', color=color)
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            return {
                'classification': classification,
                'proba_nok': float(proba),
                'proba_ok': float(1 - proba),
                'is_nok': bool(is_nok),
                'heatmap': heatmap_resized,
                'superimposed': superimposed,
                'original': img_uint8,
                'conv_layer_used': target_layer_name
            }
            
        except Exception as e:
            print(f"⚠️  Error generando Grad-CAM: {e}")
            import traceback
            traceback.print_exc()
            return {
                'classification': classification,
                'proba_nok': float(proba),
                'error': str(e)
            }
    
    def plot_training_history(self, save_path='training_history.png'):
        """Gráficas completas de entrenamiento"""
        if self.history is None:
            print("⚠️ No hay historial")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Loss & Accuracy
        axes[0,0].plot(self.history.history['loss'], label='Train')
        axes[0,0].plot(self.history.history['val_loss'], label='Val')
        axes[0,0].set_title('Loss'); axes[0,0].legend(); axes[0,0].grid()
        
        axes[0,1].plot(self.history.history['accuracy'], label='Train')
        axes[0,1].plot(self.history.history['val_accuracy'], label='Val')
        axes[0,1].set_title('Accuracy'); axes[0,1].legend(); axes[0,1].grid()
        
        # Precision & Recall
        axes[1,0].plot(self.history.history['precision'], label='Train')
        axes[1,0].plot(self.history.history['val_precision'], label='Val')
        axes[1,0].set_title('Precision'); axes[1,0].legend(); axes[1,0].grid()
        
        axes[1,1].plot(self.history.history['recall'], label='Train')
        axes[1,1].plot(self.history.history['val_recall'], label='Val')
        axes[1,1].set_title('Recall'); axes[1,1].legend(); axes[1,1].grid()
        
        # AUC & Errors
        axes[2,0].plot(self.history.history['auc'], label='Train')
        axes[2,0].plot(self.history.history['val_auc'], label='Val')
        axes[2,0].set_title('AUC'); axes[2,0].legend(); axes[2,0].grid()
        
        axes[2,1].plot(self.history.history['fp'], label='FP Train', color='orange')
        axes[2,1].plot(self.history.history['val_fp'], label='FP Val', color='red')
        axes[2,1].plot(self.history.history['fn'], label='FN Train', color='cyan')
        axes[2,1].plot(self.history.history['val_fn'], label='FN Val', color='blue')
        axes[2,1].set_title('Errors (FP & FN)'); axes[2,1].legend(); axes[2,1].grid()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✅ Gráficas guardadas: {save_path}")
        plt.close()
    
    def save_model(self, filepath='efficientnet_pill_model.pkl'):
        """Guarda modelo y metadata"""
        self.model.save(filepath.replace('.pkl', '.keras'))
        
        meta = {
            'img_size': self.img_size,
            'optimal_threshold': self.optimal_threshold,
            'trained': True,
            'class_names': self.class_names,
            'apply_dynamic_mask': self.apply_dynamic_mask
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(meta, f)
        
        print(f"\n✅ Modelo guardado:")
        print(f"   {filepath}")
        print(f"   {filepath.replace('.pkl', '.keras')}")
        print(f"   Umbral óptimo: {self.optimal_threshold:.4f}")
    
    def load_model(self, filepath='efficientnet_pill_model.pkl'):
        """Carga modelo entrenado"""
        with open(filepath, 'rb') as f:
            meta = pickle.load(f)
        
        self.img_size = meta['img_size']
        self.optimal_threshold = meta['optimal_threshold']
        self.apply_dynamic_mask = meta.get('apply_dynamic_mask', True)
        self.trained = True
        self.class_names = meta['class_names']
        
        self.model = keras.models.load_model(filepath.replace('.pkl', '.keras'))
        
        print(f"✅ Modelo cargado desde: {filepath}")
        print(f"   Umbral óptimo: {self.optimal_threshold:.4f}")
        print(f"   Máscara dinámica: {'✅' if self.apply_dynamic_mask else '❌'}")
        return self


# ====== EJEMPLO DE USO ======

if __name__ == "__main__":
    
    # CONFIGURACIÓN - Ajusta estas rutas
    OK_FOLDER = "/Users/luke/Desktop/ELADIET/dataset/crops/images/"
    NOK_FOLDER = "/Users/luke/Desktop/ELADIET/dataset/crops/NOOK/"
    
    print("="*80)
    print("🚀 EFFICIENTNET CLASSIFIER - VERSIÓN HÍBRIDA")
    print("   ✅ Recentrado geométrico robusto (Otsu + Canny)")
    print("   ✅ Máscara gaussiana adaptativa")
    print("   ✅ Manejo de desbalance (5000 OK vs 400 NOK)")
    print("   ✅ Data augmentation inteligente")
    print("   ✅ Class weights automáticos")
    print("="*80)
    
    # Crear clasificador
    classifier = EfficientNetPillClassifier(
        img_size=(300, 300),
        apply_dynamic_mask=True  # Cambiar a False para comparar
    )
    
    # Entrenar con configuración óptima para desbalance
    classifier.train(
        ok_folder=OK_FOLDER,
        nok_folder=NOK_FOLDER,
        epochs_phase1=15,      # Phase 1: Solo cabezal
        epochs_phase2=35,      # Phase 2: Fine-tuning
        batch_size=32,
        use_class_weights=True,    # ✅ CRÍTICO para 5000:400
        augment_minority=True      # ✅ Augmenta NOK hasta ratio aceptable
    )
    
    # Guardar modelo
    classifier.save_model('efficientnet_pill_hybrid.pkl')
    
    # Generar gráficas
    classifier.plot_training_history('training_history_hybrid.png')
    
    print("\n" + "="*80)
    print("🧪 TEST RÁPIDO EN IMÁGENES DE MUESTRA")
    print("="*80)
    
    # Test en algunas OK
    ok_test_paths = list(Path(OK_FOLDER).glob("*.jpg"))[:10]
    if ok_test_paths:
        print(f"\n✅ Testando {len(ok_test_paths)} imágenes OK:")
        correct_ok = 0
        for img_path in ok_test_paths:
            result = classifier.predict_single(str(img_path))
            emoji = "✅" if result['classification'] == 'OK' else "❌"
            if result['classification'] == 'OK':
                correct_ok += 1
            print(f"{emoji} {img_path.name}: {result['classification']} "
                  f"(P(NOK): {result['proba_nok']:.3f})")
        print(f"   Accuracy OK: {correct_ok}/{len(ok_test_paths)} ({correct_ok/len(ok_test_paths)*100:.1f}%)")
    
    # Test en algunas NOK
    nok_test_paths = list(Path(NOK_FOLDER).glob("*.jpg"))[:10]
    if nok_test_paths:
        print(f"\n❌ Testando {len(nok_test_paths)} imágenes NOK:")
        correct_nok = 0
        for img_path in nok_test_paths:
            result = classifier.predict_single(str(img_path))
            emoji = "✅" if result['classification'] == 'NOK' else "❌"
            if result['classification'] == 'NOK':
                correct_nok += 1
            print(f"{emoji} {img_path.name}: {result['classification']} "
                  f"(P(NOK): {result['proba_nok']:.3f})")
        print(f"   Accuracy NOK: {correct_nok}/{len(nok_test_paths)} ({correct_nok/len(nok_test_paths)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"\nArchivos generados:")
    print(f"   📄 efficientnet_pill_hybrid.pkl (metadata)")
    print(f"   🧠 efficientnet_pill_hybrid.keras (modelo)")
    print(f"   📊 training_history_hybrid.png (gráficas)")
    print(f"\n💡 Para cargar el modelo:")
    print(f"   clf = EfficientNetPillClassifier()")
    print(f"   clf.load_model('efficientnet_pill_hybrid.pkl')")
    print(f"   result = clf.predict_single('test_image.jpg')")
    print("="*80)