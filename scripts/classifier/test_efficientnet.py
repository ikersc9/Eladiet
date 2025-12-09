#!/usr/bin/env python3
"""
Script para probar el clasificador EfficientNet con visualización completa del pipeline
VERSIÓN MEJORADA: Visualiza Original + Recentrado+Máscara + Grad-CAM
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def visualize_preprocessing_steps(classifier, image_path, save_path=None):
    """
    Visualiza las etapas del preprocesamiento:
    1. Imagen original
    2. Después de recentrado
    3. Después de máscara dinámica
    4. Grad-CAM sobre la imagen procesada
    """
    
    print(f"\n🎨 Generando visualización completa del pipeline...")
    
    # Cargar imagen original
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"   ❌ No se pudo cargar la imagen")
        return None
    
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, classifier.img_size)
    
    # PASO 1: Recentrado
    img_centered, contour = classifier._center_pill_in_image(img_resized)
    
    # PASO 2: Máscara dinámica (si está activada)
    img_masked = img_centered.astype(np.float32)
    if classifier.apply_dynamic_mask and contour is not None:
        img_masked = classifier._apply_dynamic_mask(img_masked, contour, strength=0.5)
    
    # PASO 3: Sanitización
    img_final = np.nan_to_num(img_masked, nan=0.0, posinf=255.0, neginf=0.0)
    img_final = np.clip(img_final, 0, 255)
    
    # PASO 4: Grad-CAM
    gradcam_result = classifier.visualize_defect_gradcam(image_path, save_path=None)
    
    # Crear visualización combinada
    fig = plt.figure(figsize=(20, 5))
    
    # 1. Original redimensionada
    ax1 = plt.subplot(1, 4, 1)
    ax1.imshow(img_resized.astype(np.uint8))
    ax1.set_title('1. Original\n(Redimensionada)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Dibujar contorno detectado si existe
    if contour is not None:
        contour_vis = img_resized.copy()
        cv2.drawContours(contour_vis, [contour], -1, (255, 0, 0), 2)
        
        # Calcular centroide
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(contour_vis, (cX, cY), 5, (0, 255, 0), -1)
        
        ax1.imshow(contour_vis.astype(np.uint8))
        ax1.set_title('1. Original + Contorno\n(Detectado)', fontsize=12, fontweight='bold', color='blue')
    
    # 2. Recentrada
    ax2 = plt.subplot(1, 4, 2)
    ax2.imshow(img_centered.astype(np.uint8))
    ax2.set_title('2. Recentrada\n(Centroide al centro)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Marcar el centro
    h, w = img_centered.shape[:2]
    center_y, center_x = h // 2, w // 2
    img_centered_marked = img_centered.copy().astype(np.uint8)
    cv2.drawMarker(img_centered_marked, (center_x, center_y), 
                   (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    ax2.imshow(img_centered_marked)
    
    # 3. Con máscara dinámica
    ax3 = plt.subplot(1, 4, 3)
    ax3.imshow(img_final.astype(np.uint8))
    mask_status = "Activa" if classifier.apply_dynamic_mask else "Desactivada"
    ax3.set_title(f'3. Con Máscara\n({mask_status})', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Grad-CAM
    ax4 = plt.subplot(1, 4, 4)
    if 'superimposed' in gradcam_result:
        ax4.imshow(gradcam_result['superimposed'])
        classification = gradcam_result['classification']
        proba_nok = gradcam_result['proba_nok']
        color = 'red' if gradcam_result['is_nok'] else 'green'
        ax4.set_title(f'4. Grad-CAM\n{classification}: P(NOK)={proba_nok:.1%}', 
                     fontsize=12, fontweight='bold', color=color)
    else:
        ax4.text(0.5, 0.5, 'Grad-CAM\nNo disponible', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('4. Grad-CAM\n(Error)', fontsize=12, fontweight='bold', color='red')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Visualización guardada: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'original': img_resized,
        'centered': img_centered,
        'masked': img_final,
        'gradcam': gradcam_result
    }


def test_efficientnet_single(model_path, image_path, debug=False):
    """
    Prueba el clasificador con una imagen
    MEJORADO: Incluye visualización completa del pipeline
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
        print(f"   🎯 Umbral óptimo: {classifier.optimal_threshold:.4f}")
        print(f"   🎭 Máscara dinámica: {'✅ Activa' if classifier.apply_dynamic_mask else '❌ Desactivada'}")
        print(f"   🏷️  Clases: {classifier.class_names}")
        
        if debug:
            print(f"\n   🔍 DEBUG INFO:")
            print(f"      Modelo entrenado: {classifier.trained}")
            print(f"      Modelo keras path: {model_path.replace('.pkl', '.keras')}")
        
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
    proba_nok = result['proba_nok']
    proba_ok = result['proba_ok']
    is_nok = result['is_nok']
    threshold_used = result.get('threshold_used', 0.5)
    raw_score = result.get('raw_score', proba_nok)
    
    # Emoji según resultado
    emoji = "❌" if is_nok else "✅"
    
    print(f"\n{emoji} CLASIFICACIÓN: {classification}")
    print(f"📊 P(NOK): {proba_nok:.2%}")
    print(f"📊 P(OK): {proba_ok:.2%}")
    print(f"🎯 Score raw: {raw_score:.4f}")
    print(f"🎯 Threshold usado: {threshold_used:.4f}")
    print(f"📏 Distancia al threshold: {abs(raw_score - threshold_used):.4f}")
    
    print(f"\n💡 Interpretación:")
    if is_nok:
        print(f"   • La pastilla es DEFECTUOSA")
        print(f"   • Probabilidad de defecto: {proba_nok:.1%}")
        print(f"   • Supera el threshold de {threshold_used:.2%}")
        if proba_nok > 0.9:
            print(f"   • ⚠️  Defecto MUY EVIDENTE (>90%)")
        elif proba_nok > 0.7:
            print(f"   • ⚠️  Defecto CLARO (>70%)")
        else:
            print(f"   • ⚠️  Defecto POSIBLE ({threshold_used*100:.0f}-70%)")
    else:
        print(f"   • La pastilla es NORMAL")
        print(f"   • Probabilidad de estar OK: {proba_ok:.1%}")
        print(f"   • Por debajo del threshold de {threshold_used:.2%}")
        if proba_ok > 0.95:
            print(f"   • ✅ Pastilla CLARAMENTE OK (>95%)")
        elif proba_ok > 0.8:
            print(f"   • ✅ Pastilla OK (>80%)")
        else:
            print(f"   • ⚠️  Pastilla OK pero cerca del límite")
    
    # Análisis de confianza
    confidence_distance = abs(raw_score - threshold_used)
    print(f"\n🎯 Análisis de Certeza:")
    if confidence_distance > 0.3:
        print(f"   ✅ Certeza ALTA (distancia al threshold: {confidence_distance:.3f})")
        print(f"      El modelo está muy seguro de su predicción")
    elif confidence_distance > 0.15:
        print(f"   ⚠️  Certeza MEDIA (distancia al threshold: {confidence_distance:.3f})")
        print(f"      El modelo tiene cierta incertidumbre")
    else:
        print(f"   ⚠️  Certeza BAJA (distancia al threshold: {confidence_distance:.3f})")
        print(f"      Predicción muy cerca del límite - revisar manualmente")
    
    print("\n" + "="*80)
    
    # 6. VISUALIZACIÓN COMPLETA DEL PIPELINE
    print(f"\n5️⃣ Generando visualización completa del pipeline...")
    
    try:
        import os
        vis_dir = "pipeline_visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        vis_save_path = f"{vis_dir}/{Path(image_path).stem}_pipeline_complete.png"
        
        vis_result = visualize_preprocessing_steps(
            classifier=classifier,
            image_path=image_path,
            save_path=vis_save_path
        )
        
        if vis_result:
            print(f"   ✅ Visualización completa guardada!")
            print(f"      📁 {vis_save_path}")
            print(f"\n   📊 Etapas visualizadas:")
            print(f"      1. Original + Contorno detectado")
            print(f"      2. Recentrada (centroide al centro)")
            print(f"      3. Con máscara dinámica aplicada")
            print(f"      4. Grad-CAM (zonas de atención del modelo)")
        else:
            print(f"   ⚠️  No se pudo generar visualización completa")
            
    except Exception as e:
        print(f"   ⚠️  Error en visualización: {e}")
        import traceback
        traceback.print_exc()
        
    if debug:
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
    CLASSIFIER_MODEL = "/Users/luke/Desktop/ELADIET/efficientnet_pill_hybrid.pkl"
    
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
            if hasattr(system.classifier, 'efficientnet_classifier'):
                eff_clf = system.classifier.efficientnet_classifier
                print(f"   Umbral óptimo: {eff_clf.optimal_threshold:.4f}")
                print(f"   Máscara dinámica: {'✅' if eff_clf.apply_dynamic_mask else '❌'}")
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
    
    from efficientnet_classifier_hybrid import EfficientNetPillClassifier
    classifier = EfficientNetPillClassifier()
    classifier.load_model(model_path)
    
    print(f"🎯 Umbral óptimo del modelo: {classifier.optimal_threshold:.4f}")
    print(f"🎭 Máscara dinámica: {'✅ Activa' if classifier.apply_dynamic_mask else '❌ Desactivada'}\n")
    
    results_ok = []
    results_nok = []
    
    # Procesar imágenes
    for img_path in images[:50]:  # Primeras 50
        result = classifier.predict_single(str(img_path))
        
        if 'error' not in result:
            classification = result['classification']
            proba_nok = result['proba_nok']
            raw_score = result.get('raw_score', proba_nok)
            
            emoji = "✅" if classification == 'OK' else "❌"
            print(f"{emoji} {img_path.name:40s} → {classification:3s} "
                  f"(score: {raw_score:.4f}, P(NOK): {proba_nok:.3f})")
            
            if classification == 'OK':
                results_ok.append((img_path.name, raw_score, proba_nok))
            else:
                results_nok.append((img_path.name, raw_score, proba_nok))
    
    # Resumen
    print("\n" + "="*80)
    print("📊 RESUMEN")
    print("="*80)
    
    total = len(results_ok) + len(results_nok)
    print(f"\n✅ OK:  {len(results_ok)} ({len(results_ok)/total*100:.1f}%)")
    print(f"❌ NOK: {len(results_nok)} ({len(results_nok)/total*100:.1f}%)")
    print(f"📊 Total: {total}")
    
    if results_ok:
        print(f"\n🟢 Top 3 más claramente OK (score más bajo):")
        for name, score, pnok in sorted(results_ok, key=lambda x: x[1])[:3]:
            print(f"   • {name}: score={score:.4f} (P(NOK): {pnok:.3f})")
    
    if results_nok:
        print(f"\n🔴 Top 3 más claramente NOK (score más alto):")
        for name, score, pnok in sorted(results_nok, key=lambda x: x[1], reverse=True)[:3]:
            print(f"   • {name}: score={score:.4f} (P(NOK): {pnok:.3f})")
    
    print("\n" + "="*80 + "\n")


def evaluate_on_labeled_data(model_path, ok_folder, nok_folder):
    """Evalúa el modelo en datos etiquetados"""
    
    print("\n" + "="*80)
    print("📊 EVALUACIÓN EN DATOS ETIQUETADOS")
    print("="*80)
    
    from efficientnet_classifier_hybrid import EfficientNetPillClassifier
    classifier = EfficientNetPillClassifier()
    classifier.load_model(model_path)
    
    print(f"\n🎯 Threshold del modelo: {classifier.optimal_threshold:.4f}")
    print(f"🎭 Máscara dinámica: {'✅ Activa' if classifier.apply_dynamic_mask else '❌ Desactivada'}")
    
    # Evaluar OK
    ok_path = Path(ok_folder)
    if ok_path.exists():
        print(f"\n📁 Evaluando carpeta OK: {ok_folder}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in ok_path.iterdir() if f.suffix.lower() in image_extensions][:100]
        
        correct = 0
        total = len(images)
        scores = []
        
        for img in images:
            result = classifier.predict_single(str(img))
            if result['classification'] == 'OK':
                correct += 1
            scores.append(result.get('raw_score', result['proba_nok']))
        
        accuracy = correct / total if total > 0 else 0
        mean_score = np.mean(scores) if scores else 0
        
        print(f"   ✅ Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
        print(f"   📊 Score medio: {mean_score:.4f}")
        print(f"   📊 Score std: {np.std(scores):.4f}")
        
        if accuracy < 0.90:
            print(f"   ⚠️  ADVERTENCIA: Accuracy baja (<90%)")
            print(f"      Muchos falsos positivos (OK clasificadas como NOK)")
    
    # Evaluar NOK
    nok_path = Path(nok_folder)
    if nok_path.exists():
        print(f"\n📁 Evaluando carpeta NOK: {nok_folder}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in nok_path.iterdir() if f.suffix.lower() in image_extensions][:100]
        
        correct = 0
        total = len(images)
        scores = []
        
        for img in images:
            result = classifier.predict_single(str(img))
            if result['classification'] == 'NOK':
                correct += 1
            scores.append(result.get('raw_score', result['proba_nok']))
        
        recall = correct / total if total > 0 else 0
        mean_score = np.mean(scores) if scores else 0
        
        print(f"   ❌ Recall: {recall*100:.1f}% ({correct}/{total})")
        print(f"   📊 Score medio: {mean_score:.4f}")
        print(f"   📊 Score std: {np.std(scores):.4f}")
        
        if recall < 0.80:
            print(f"   ⚠️  ADVERTENCIA: Recall bajo (<80%)")
            print(f"      Muchos falsos negativos (NOK clasificadas como OK)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    
    # Configuración por defecto
    DEFAULT_MODEL = "/Users/luke/Desktop/ELADIET/efficientnet_pill_hybrid.pkl"
    
    print("\n" + "="*80)
    print("🔬 TEST DEL CLASIFICADOR EFFICIENTNET - VERSIÓN HÍBRIDA")
    print("   ✅ Recentrado geométrico (Otsu + Canny)")
    print("   ✅ Máscara gaussiana adaptativa")
    print("   ✅ Visualización completa del pipeline")
    print("="*80)
    
    # Uso
    if len(sys.argv) == 1:
        print("\n💡 Uso:")
        print(f"   python3 {sys.argv[0]} <imagen.jpg>              # Test con visualización completa")
        print(f"   python3 {sys.argv[0]} --pipeline <imagen.jpg>   # Test del pipeline completo")
        print(f"   python3 {sys.argv[0]} <carpeta/>                # Test múltiples imágenes")
        print(f"   python3 {sys.argv[0]} --debug <imagen.jpg>      # Test con debug info")
        print(f"   python3 {sys.argv[0]} eval <ok_folder> <nok_folder>  # Evaluar en datos etiquetados")
        print(f"\nEjemplos:")
        print(f"   python3 {sys.argv[0]} test5.png")
        print(f"   python3 {sys.argv[0]} --pipeline test5.png")
        print(f"   python3 {sys.argv[0]} --debug test5.png")
        print(f"   python3 {sys.argv[0]} crops_ok/")
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
        # Test individual con visualización completa
        test_efficientnet_single(model_path, input_path, debug=debug)
    
    elif path.is_dir():
        # Test múltiple
        test_multiple_images(model_path, input_path)
    
    else:
        print(f"\n❌ No se encontró: {input_path}")
        print("   Especifica una imagen o carpeta válida")
        sys.exit(1)