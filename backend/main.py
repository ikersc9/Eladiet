from ultralytics import YOLO
import cv2
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras

class AutoencoderClassifier:
    """Clasificador de anomalías basado en Autoencoder"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.threshold = None
        self.img_size = None
        self.load_model()
    
    def load_model(self):
        """Carga el modelo de Autoencoder"""
        try:
            # Cargar metadata
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.img_size = model_data['img_size']
            self.threshold = model_data['threshold']
            
            # Cargar modelo de Keras
            model_keras_path = self.model_path.replace('.pkl', '_model.keras')
            self.model = keras.models.load_model(model_keras_path)
            
            print(f"✅ Clasificador Autoencoder cargado correctamente")
            print(f"   Threshold: {self.threshold:.6f}")
        except Exception as e:
            print(f"⚠️  Error al cargar clasificador: {e}")
            raise
    
    def preprocess_image(self, img):
        """Preprocesa la imagen para el autoencoder"""
        if img is None or img.size == 0:
            return None
        
        # Convertir BGR a RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        img_resized = cv2.resize(img_rgb, self.img_size)
        
        # Normalizar a [0, 1]
        img_normalized = img_resized / 255.0
        
        return img_normalized
    
    def predict(self, img):
        """
        Predice si la imagen es OK o NOK
        
        Returns:
            tuple: (classification, confidence, error)
                   classification: 'OK' o 'NOK'
                   confidence: float (0-1)
                   error: error de reconstrucción
        """
        # Preprocesar
        img_processed = self.preprocess_image(img)
        if img_processed is None:
            return 'UNKNOWN', 0.0, 0.0
        
        # Predecir
        img_batch = np.expand_dims(img_processed, axis=0)
        reconstruction = self.model.predict(img_batch, verbose=0)[0]
        
        # Calcular error de reconstrucción
        error = np.mean(np.square(img_processed - reconstruction))
        
        # Clasificar
        is_anomaly = error > self.threshold
        classification = 'NOK' if is_anomaly else 'OK'
        
        # Confianza normalizada (basada en cuánto excede o está debajo del threshold)
        if is_anomaly:
            # Para NOK: cuanto mayor el error, mayor la confianza
            confidence = min(error / self.threshold, 2.0) / 2.0  # Normalizar a [0.5, 1.0]
            confidence = 0.5 + (confidence * 0.5)
        else:
            # Para OK: cuanto menor el error, mayor la confianza
            confidence = 1.0 - min(error / self.threshold, 1.0)
            confidence = max(confidence, 0.5)  # Mínimo 50% de confianza
        
        return classification, float(confidence), float(error)

class PillDetectionSystem:
    def __init__(self, detector_model_path: str, classifier_model_path: str = None):
        """
        Sistema completo de detección y clasificación de pastillas
        
        Args:
            detector_model_path: Ruta al modelo YOLO de detección
            classifier_model_path: Ruta al modelo Autoencoder (.pkl)
        """
        self.detector = YOLO(detector_model_path)
        self.classifier = None
        
        if classifier_model_path and os.path.exists(classifier_model_path):
            try:
                self.classifier = AutoencoderClassifier(classifier_model_path)
                print("✅ Sistema inicializado con detector y clasificador")
            except Exception as e:
                print(f"⚠️  Clasificador no disponible: {e}")
                print("   El sistema funcionará solo con detección")
        else:
            print("✅ Sistema inicializado solo con detector")
            if classifier_model_path:
                print(f"   Clasificador no encontrado en: {classifier_model_path}")
        
    def process_images(self, image_paths: List[str], output_base_dir: str = "output") -> Dict:
        """
        Procesa una o múltiples imágenes, detecta pastillas y las clasifica
        
        Args:
            image_paths: Lista de rutas a imágenes
            output_base_dir: Directorio base para guardar resultados
            
        Returns:
            Dict con metadata completa del procesamiento
        """
        # Crear estructura de directorios
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(output_base_dir) / f"session_{timestamp}"
        crops_dir = session_dir / "crops"
        annotated_dir = session_dir / "annotated"
        
        crops_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata para reconstrucción
        session_metadata = {
            "timestamp": timestamp,
            "images": []
        }
        
        print("\n" + "="*80)
        print("🔬 ELADIET - Sistema de Detección de Pastillas")
        print("="*80)
        
        for img_path in image_paths:
            print(f"\n📸 Procesando imagen: {Path(img_path).name}")
            print("-" * 80)
            result = self._process_single_image(
                img_path, 
                crops_dir, 
                annotated_dir
            )
            session_metadata["images"].append(result)
        
        # Guardar metadata
        metadata_path = session_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(session_metadata, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*80)
        print(f"✅ Procesamiento completado!")
        print(f"📁 Resultados guardados en: {session_dir}")
        print(f"📄 Metadata: {metadata_path}")
        print("="*80 + "\n")
        
        return session_metadata
    
    def _process_single_image(
        self, 
        img_path: str, 
        crops_dir: Path, 
        annotated_dir: Path
    ) -> Dict:
        """Procesa una imagen individual"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {img_path}")
        
        img_name = Path(img_path).stem
        original_filename = Path(img_path).name
        
        # Detectar pastillas
        detection_results = self.detector(img)
        
        image_metadata = {
            "original_path": img_path,
            "original_filename": original_filename,
            "image_name": img_name,
            "pills": []
        }
        
        # Procesar cada detección
        pill_count = 0
        for idx, result in enumerate(detection_results):
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for pill_idx, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                
                # Extraer crop
                crop_bgr = img[y1:y2, x1:x2]
                
                # Clasificar si hay clasificador
                classification = "UNKNOWN"
                class_confidence = 0.0
                reconstruction_error = 0.0
                
                if self.classifier and crop_bgr.size > 0:
                    try:
                        classification, class_confidence, reconstruction_error = self.classifier.predict(crop_bgr)
                    except Exception as e:
                        print(f"     ⚠️  Error al clasificar: {e}")
                        classification = "UNKNOWN"
                        class_confidence = 0.0
                
                # Guardar crop
                crop_filename = f"{img_name}_pill_{pill_count:03d}.jpg"
                crop_path = crops_dir / crop_filename
                cv2.imwrite(str(crop_path), crop_bgr)
                
                # Metadata del crop
                pill_data = {
                    "pill_id": pill_count,
                    "crop_filename": crop_filename,
                    "bbox": [x1, y1, x2, y2],
                    "detection_confidence": float(conf),
                    "classification": classification,
                    "classification_confidence": float(class_confidence),
                    "reconstruction_error": float(reconstruction_error) if self.classifier else None
                }
                
                image_metadata["pills"].append(pill_data)
                
                # Print mejorado con nombre de imagen original y crop
                status_icon = "❌" if classification == "NOK" else "✅" if classification == "OK" else "❓"
                print(f"  {status_icon} Pastilla #{pill_count + 1:02d}")
                print(f"     └─ Imagen original: {original_filename}")
                print(f"     └─ Crop guardado: {crop_filename}")
                print(f"     └─ Clasificación: {classification} (confianza: {class_confidence:.2%})")
                if self.classifier and reconstruction_error > 0:
                    print(f"     └─ Error reconstrucción: {reconstruction_error:.6f}")
                print(f"     └─ Detección: {conf:.2%}")
                
                pill_count += 1
        
        # Crear imagen anotada
        annotated_img = self._create_annotated_image(img, image_metadata["pills"])
        annotated_path = annotated_dir / f"{img_name}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_img)
        
        image_metadata["total_pills"] = len(image_metadata["pills"])
        image_metadata["annotated_path"] = str(annotated_path)
        
        print(f"\n  📊 Total de pastillas detectadas: {pill_count}")
        if pill_count > 0:
            ok_count = sum(1 for p in image_metadata['pills'] if p['classification'] == 'OK')
            nok_count = sum(1 for p in image_metadata['pills'] if p['classification'] == 'NOK')
            unknown_count = pill_count - ok_count - nok_count
            print(f"  ✅ OK: {ok_count} | ❌ NOK: {nok_count} | ❓ Sin clasificar: {unknown_count}")
        
        return image_metadata
    
    def _create_annotated_image(self, img: np.ndarray, pills: List[Dict]) -> np.ndarray:
        """Crea imagen con anotaciones de detección y clasificación"""
        annotated = img.copy()
        
        for pill in pills:
            x1, y1, x2, y2 = pill["bbox"]
            classification = pill["classification"]
            conf = pill["classification_confidence"]
            pill_id = pill["pill_id"]
            
            # Color según clasificación
            if classification == "NOK":
                color = (0, 0, 255)  # Rojo
                thickness = 3
            elif classification == "OK":
                color = (0, 255, 0)  # Verde
                thickness = 2
            else:
                color = (255, 165, 0)  # Naranja (UNKNOWN)
                thickness = 2
            
            # Dibujar bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Etiqueta con ID
            label = f"#{pill_id+1} {classification}"
            if conf > 0:
                label += f" {conf:.0%}"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Fondo para el texto
            cv2.rectangle(
                annotated, 
                (x1, y1 - label_size[1] - 10), 
                (x1 + label_size[0], y1), 
                color, 
                -1
            )
            
            # Texto
            cv2.putText(
                annotated, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
        
        return annotated