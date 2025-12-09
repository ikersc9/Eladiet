#!/usr/bin/env python3
"""
Script para probar el clasificador EfficientNet con imágenes individuales o carpetas
MEJORADO: Incluye test del pipeline completo y debugging
"""

import sys
import cv2
import numpy as np
from pathlib import Path


def test_efficientnet_single(model_path, image_path, debug=False):
    """
    Prueba el clasificador con una imagen
    
    Args:
        model_path: Ruta al modelo .pkl
        image_path: Ruta a la imagen
        debug: Si True, muestra información extra de debugging
    """
    
    print("\n" + "="*80)
    print("🧪 TEST DEL CLASIFICADOR EFFICIENTNET (Modo Standalone)")
    print("="*80)
    
    # 1. Verificar archivos
    print("\n1️⃣ Verificando archivos...")
    
    if not Path(model_path).exists():
        print(f"   ❌ Modelo no encontrado: {model_path}")
        return
    print(f"   ✅ Modelo encontrado: {Path(model_path).name}")
    
    if not Path(image_path).exists():
        print(f"   ❌ Imagen no encontrada: {image_path}")
        return
    print(f"   ✅ Imagen encontrada: {Path(image_path).name}")
    
    # 2. Cargar clasificador
    print("\n2️⃣ Cargando clasificador...")
    try:
        from efficientnet_classifier import EfficientNetPillClassifier
        
        classifier = EfficientNetPillClassifier()
        classifier.load_model(model_path)
        
        print(f"   ✅ Clasificador cargado correctamente")
        print(f"   📏 Tamaño de imagen: {classifier.img_size}")
        print(f"   🏷️  Clases: {classifier.class_names}")
        
        if debug:
            print(f"\n   🔍 DEBUG INFO:")
            print(f"      Modelo entrenado: {classifier.trained}")
            print(f"      Modelo keras path: {model_path.replace('.pkl', '_model.keras')}")
        
    except Exception as e:
        print(f"   ❌ Error al cargar clasificador: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Cargar imagen
    print(f"\n3️⃣ Cargando imagen...")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"   ❌ No se pudo leer la imagen")
        return
    
    print(f"   ✅ Imagen cargada: {img.shape}")
    
    if debug:
        print(f"\n   🔍 DEBUG INFO:")
        print(f"      Formato: BGR (OpenCV)")
        print(f"      Dtype: {img.dtype}")
        print(f"      Min/Max values: {img.min()}/{img.max()}")
    
    # 4. Predecir
    print(f"\n4️⃣ Ejecutando predicción...")
    try:
        result = classifier.predict_single(image_path)
        
        if 'error' in result:
            print(f"   ❌ Error en predicción: {result['error']}")
            return
        
        print(f"   ✅ Predicción completada!")
        
    except Exception as e:
        print(f"   ❌ Error durante predicción: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Mostrar resultados
    print("\n" + "="*80)
    print("📊 RESULTADOS DE LA CLASIFICACIÓN")
    print("="*80)
    
    classification = result['classification']
    confidence = result['confidence']
    proba_nok = result['proba_nok']
    proba_ok = result['proba_ok']
    is_nok = result['is_nok']
    threshold_used = result.get('threshold_used', 0.5)
    
    # Emoji según resultado
    emoji = "❌" if is_nok else "✅"
    
    print(f"\n{emoji} CLASIFICACIÓN: {classification}")
    print(f"📈 CONFIANZA: {confidence:.2%}")
    print(f"📊 P(NOK): {proba_nok:.2%}")
    print(f"📊 P(OK): {proba_ok:.2%}")
    print(f"🎯 Threshold usado: {threshold_used:.2f}")
    
    print(f"\n💡 Interpretación:")
    if is_nok:
        print(f"   • La pastilla es DEFECTUOSA")
        print(f"   • Probabilidad de defecto: {proba_nok:.1%}")
        print(f"   • Supera el threshold de {threshold_used:.0%}")
        if proba_nok > 0.9:
            print(f"   • ⚠️  Defecto MUY EVIDENTE (>90%)")
        elif proba_nok > 0.7:
            print(f"   • ⚠️  Defecto CLARO (>70%)")
        else:
            print(f"   • ⚠️  Defecto POSIBLE ({threshold_used*100:.0f}-70%)")
    else:
        print(f"   • La pastilla es NORMAL")
        print(f"   • Probabilidad de estar OK: {proba_ok:.1%}")
        print(f"   • Por debajo del threshold de {threshold_used:.0%}")
        if proba_ok > 0.95:
            print(f"   • ✅ Pastilla CLARAMENTE OK (>95%)")
        elif proba_ok > 0.8:
            print(f"   • ✅ Pastilla OK (>80%)")
        else:
            print(f"   • ⚠️  Pastilla OK pero cerca del límite ({threshold_used*100:.0f}-80%)")
    
    # Análisis de confianza
    print(f"\n🎯 Análisis de Confianza:")
    if confidence > 0.85:
        print(f"   ✅ Confianza ALTA (>{confidence:.0%})")
        print(f"      El modelo está muy seguro de su predicción")
    elif confidence > 0.70:
        print(f"   ⚠️  Confianza MEDIA ({confidence:.0%})")
        print(f"      El modelo tiene cierta incertidumbre")
    else:
        print(f"   ⚠️  Confianza BAJA (<{confidence:.0%})")
        print(f"      Revisar manualmente recomendado")
    
    print("\n" + "="*80)
    
    # Generar visualización Grad-CAM automática
    try:
        # Importante: crear carpeta si no existe
        vis_dir = "gradcam_visualizations"
        import os
        os.makedirs(vis_dir, exist_ok=True)
        
        vis_save_path = f"{vis_dir}/{Path(image_path).stem}_gradcam.png"
        print(f"🎨 Generando visualización Grad-CAM: {vis_save_path}")
        vis_result = classifier.visualize_defect_gradcam(
            image_path=image_path,
            save_path=vis_save_path
        )
        if 'error' in vis_result:
            print(f"   ⚠️  Error en visualización: {vis_result['error']}")
        else:
            print(f"   ✅ Visualización guardada correctamente")
    except Exception as e:
        print(f"   ⚠️  No se pudo generar visualización: {e}")
        
    if debug:
        # Mostrar info del resultado completo
        print("\n🔍 DEBUG - Resultado completo (dict):")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.6f}")
            else:
                print(f"   {key}: {value}")
        print("\n" + "="*80)
    
    print()
    
    return result


def test_full_pipeline(image_path, debug=False):
    """
    Prueba el pipeline completo: detector YOLO + clasificador EfficientNet
    Simula exactamente lo que hace el servidor Flask
    """
    
    print("\n" + "="*80)
    print("🔬 TEST DEL PIPELINE COMPLETO (Detector + Clasificador)")
    print("="*80)
    
    # Configuración exacta de app.py
    DETECTOR_MODEL = "/Users/luke/Desktop/ELADIET/models/best-3.pt"
    CLASSIFIER_MODEL = "/Users/luke/Desktop/ELADIET/models/efficientnet_pill.keras"
    
    if not Path(image_path).exists():
        print(f"\n❌ Imagen no encontrada: {image_path}")
        return
    
    try:
        from main import PillDetectionSystem
        
        # 1. Inicializar sistema
        print(f"\n1️⃣ Inicializando sistema...")
        print(f"   Detector: {Path(DETECTOR_MODEL).name}")
        print(f"   Clasificador: {Path(CLASSIFIER_MODEL).name}")
        
        system = PillDetectionSystem(
            detector_model_path=DETECTOR_MODEL,
            classifier_model_path=CLASSIFIER_MODEL
        )
        
        print(f"\n   ✅ Sistema inicializado")
        if system.classifier:
            print(f"   Tipo de clasificador: {system.classifier.classifier_type}")
        else:
            print(f"   ⚠️  Sin clasificador")
        
        # 2. Procesar imagen
        print(f"\n2️⃣ Procesando imagen completa: {Path(image_path).name}")
        print("-" * 80)
        
        metadata = system.process_images(
            [image_path],
            output_base_dir="test_output"
        )
        
        # 3. Analizar resultados
        print("\n" + "="*80)
        print("📊 RESULTADOS DEL PIPELINE")
        print("="*80)
        
        for img_data in metadata['images']:
            total_pills = img_data['total_pills']
            
            if total_pills == 0:
                print(f"\n⚠️  No se detectaron pastillas en la imagen")
                return
            
            ok_count = sum(1 for p in img_data['pills'] if p['classification'] == 'OK')
            nok_count = sum(1 for p in img_data['pills'] if p['classification'] == 'NOK')
            unknown_count = sum(1 for p in img_data['pills'] if p['classification'] == 'UNKNOWN')
            
            print(f"\n📷 Imagen: {img_data['original_filename']}")
            print(f"   Total pastillas detectadas: {total_pills}")
            print(f"\n   Clasificaciones:")
            print(f"   ✅ OK: {ok_count} ({ok_count/total_pills*100:.1f}%)")
            print(f"   ❌ NOK: {nok_count} ({nok_count/total_pills*100:.1f}%)")
            print(f"   ❓ UNKNOWN: {unknown_count} ({unknown_count/total_pills*100:.1f}%)")
            
            # Verificar si hay problema
            if unknown_count == total_pills:
                print(f"\n   🚨 PROBLEMA DETECTADO: Todas las clasificaciones son UNKNOWN")
                print(f"      Esto indica que el clasificador no está funcionando en el pipeline")
                print(f"\n   Posibles causas:")
                print(f"      1. Error en la carga del modelo")
                print(f"      2. Error en el preprocesamiento de imágenes")
                print(f"      3. Excepción silenciosa en la predicción")
                
                if debug and img_data['pills']:
                    pill = img_data['pills'][0]
                    print(f"\n   🔍 DEBUG - Primer pill:")
                    print(f"      classification: {pill['classification']}")
                    print(f"      classification_confidence: {pill['classification_confidence']}")
                    print(f"      reconstruction_error: {pill.get('reconstruction_error', 'N/A')}")
            
            elif unknown_count > 0:
                print(f"\n   ⚠️  Algunas clasificaciones son UNKNOWN ({unknown_count}/{total_pills})")
            else:
                print(f"\n   ✅ Todas las pastillas clasificadas correctamente")
            
            # Detalles por pastilla
            print(f"\n   Detalles de cada pastilla:")
            for pill in img_data['pills']:
                status_icon = "✅" if pill['classification'] == 'OK' else "❌" if pill['classification'] == 'NOK' else "❓"
                conf_str = f"{pill['classification_confidence']:.1%}" if pill['classification_confidence'] > 0 else "N/A"
                print(f"   {status_icon} Pill #{pill['pill_id']+1:02d}: {pill['classification']} (conf: {conf_str})")
        
        print("\n" + "="*80)
        
        if unknown_count == 0:
            print("✅ TEST PASSED: Pipeline completo funcionando correctamente")
        elif unknown_count < total_pills:
            print("⚠️  TEST PARCIAL: Algunas clasificaciones funcionan")
        else:
            print("❌ TEST FAILED: Todas las clasificaciones son UNKNOWN")
        
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_images(model_path, images_folder):
    """Prueba el clasificador con múltiples imágenes de una carpeta"""
    
    print("\n" + "="*80)
    print(f"📁 PROBANDO MÚLTIPLES IMÁGENES: {images_folder}")
    print("="*80 + "\n")
    
    folder = Path(images_folder)
    if not folder.exists():
        print(f"❌ Carpeta no encontrada")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = sorted([f for f in folder.iterdir() if f.suffix.lower() in image_extensions])
    
    if not images:
        print(f"❌ No se encontraron imágenes en la carpeta")
        return
    
    print(f"✅ Encontradas {len(images)} imágenes\n")
    
    from efficientnet_classifier import EfficientNetPillClassifier
    classifier = EfficientNetPillClassifier()
    classifier.load_model(model_path)
    
    results_ok = []
    results_nok = []
    
    # Procesar imágenes
    for img_path in images[:50]:  # Primeras 50
        result = classifier.predict_single(str(img_path))
        
        if 'error' not in result:
            classification = result['classification']
            confidence = result['confidence']
            proba_nok = result['proba_nok']
            
            emoji = "✅" if classification == 'OK' else "❌"
            print(f"{emoji} {img_path.name:40s} → {classification:3s} (conf: {confidence:.1%}, P(NOK): {proba_nok:.1%})")
            
            if classification == 'OK':
                results_ok.append((img_path.name, confidence, proba_nok))
            else:
                results_nok.append((img_path.name, confidence, proba_nok))
    
    # Resumen
    print("\n" + "="*80)
    print("📊 RESUMEN")
    print("="*80)
    
    total = len(results_ok) + len(results_nok)
    print(f"\n✅ OK:  {len(results_ok)} ({len(results_ok)/total*100:.1f}%)")
    print(f"❌ NOK: {len(results_nok)} ({len(results_nok)/total*100:.1f}%)")
    print(f"📊 Total: {total}")
    
    if results_ok:
        print(f"\n🟢 Top 3 más confiadas como OK:")
        for name, conf, pnok in sorted(results_ok, key=lambda x: 1-x[2])[:3]:
            print(f"   • {name}: {conf:.1%} confianza (P(NOK): {pnok:.1%})")
    
    if results_nok:
        print(f"\n🔴 Top 3 más confiadas como NOK:")
        for name, conf, pnok in sorted(results_nok, key=lambda x: x[2], reverse=True)[:3]:
            print(f"   • {name}: {conf:.1%} confianza (P(NOK): {pnok:.1%})")
    
    print("\n" + "="*80 + "\n")


def evaluate_on_labeled_data(model_path, ok_folder, nok_folder):
    """Evalúa el modelo en datos etiquetados"""
    
    print("\n" + "="*80)
    print("📊 EVALUACIÓN EN DATOS ETIQUETADOS")
    print("="*80)
    
    from efficientnet_classifier import EfficientNetPillClassifier
    classifier = EfficientNetPillClassifier()
    classifier.load_model(model_path)
    
    print(f"\n🎯 Threshold del modelo: 0.5 (fijo)")
    
    # Evaluar OK
    ok_path = Path(ok_folder)
    if ok_path.exists():
        print(f"\n📁 Evaluando carpeta OK: {ok_folder}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in ok_path.iterdir() if f.suffix.lower() in image_extensions][:100]
        
        correct = 0
        total = len(images)
        
        for img in images:
            result = classifier.predict_single(str(img))
            if result['classification'] == 'OK':
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"   ✅ Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
        
        if accuracy < 0.90:
            print(f"   ⚠️  ADVERTENCIA: Accuracy baja (<90%)")
            print(f"      Muchos falsos positivos (OK como NOK)")
            print(f"      El modelo necesita ser re-entrenado con mejor balance")
    
    # Evaluar NOK
    nok_path = Path(nok_folder)
    if nok_path.exists():
        print(f"\n📁 Evaluando carpeta NOK: {nok_folder}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in nok_path.iterdir() if f.suffix.lower() in image_extensions][:100]
        
        correct = 0
        total = len(images)
        
        for img in images:
            result = classifier.predict_single(str(img))
            if result['classification'] == 'NOK':
                correct += 1
        
        recall = correct / total if total > 0 else 0
        print(f"   ❌ Recall: {recall*100:.1f}% ({correct}/{total})")
        
        if recall < 0.80:
            print(f"   ⚠️  ADVERTENCIA: Recall bajo (<80%)")
            print(f"      Muchos falsos negativos (NOK como OK)")
            print(f"      El modelo necesita ser re-entrenado con más datos NOK")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    
    # Configuración por defecto
    DEFAULT_MODEL = "/Users/luke/Desktop/ELADIET/saved_models/best_model/efficientnet_pastillas.pkl"
    #DEFAULT_MODEL = "/Users/luke/Desktop/ELADIET/saved_models/focused_model/efficientnet_pastillas_focused.pkl"
    print("\n" + "="*80)
    print("🔬 TEST DEL CLASIFICADOR EFFICIENTNET")
    print("="*80)
    
    # Uso
    if len(sys.argv) == 1:
        print("\n💡 Uso:")
        print(f"   python3 {sys.argv[0]} <imagen.jpg>              # Test standalone del clasificador")
        print(f"   python3 {sys.argv[0]} --pipeline <imagen.jpg>   # Test del pipeline completo")
        print(f"   python3 {sys.argv[0]} <carpeta/>                # Test múltiples imágenes")
        print(f"   python3 {sys.argv[0]} --debug <imagen.jpg>      # Test con debug info")
        print(f"   python3 {sys.argv[0]} eval <ok_folder> <nok_folder>  # Evaluar en datos etiquetados")
        print(f"\nEjemplos:")
        print(f"   python3 {sys.argv[0]} test5.png")
        print(f"   python3 {sys.argv[0]} --pipeline test5.png")
        print(f"   python3 {sys.argv[0]} --debug test5.png")
        print(f"   python3 {sys.argv[0]} eval crops_ok/ crops_nok/")
        print()
        sys.exit(0)
    
    # Modo debug
    debug = '--debug' in sys.argv
    if debug:
        sys.argv.remove('--debug')
    
    # Modo evaluación
    if sys.argv[1] == 'eval' and len(sys.argv) >= 4:
        ok_folder = sys.argv[2]
        nok_folder = sys.argv[3]
        model_path = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_MODEL
        
        evaluate_on_labeled_data(model_path, ok_folder, nok_folder)
        sys.exit(0)
    
    # Modo pipeline
    if '--pipeline' in sys.argv:
        sys.argv.remove('--pipeline')
        if len(sys.argv) < 2:
            print("❌ Especifica una imagen para probar el pipeline")
            sys.exit(1)
        test_full_pipeline(sys.argv[1], debug=debug)
        sys.exit(0)
    
    input_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    
    # Verificar si es archivo o carpeta
    path = Path(input_path)
    
    if path.is_file():
        test_efficientnet_single(model_path, input_path, debug=debug)
    
    elif path.is_dir():
        # Test múltiple
        test_multiple_images(model_path, input_path)
    
    else:
        print(f"\n❌ No se encontró: {input_path}")
        print("   Especifica una imagen o carpeta válida")
        sys.exit(1)
