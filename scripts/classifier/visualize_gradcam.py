#!/usr/bin/env python3
"""
Script para visualizar dónde el modelo detecta defectos usando Grad-CAM
Usa un modelo EfficientNet ya entrenado
"""

import sys
import os
from pathlib import Path
import argparse
from efficientnet_classifier import EfficientNetPillClassifier


def visualize_images_gradcam(image_paths, model_path, output_dir='gradcam_visualizations'):
    """
    Genera visualizaciones Grad-CAM para una lista de imágenes
    
    Args:
        image_paths: Lista de rutas a imágenes
        model_path: Ruta al modelo entrenado (.pkl)
        output_dir: Carpeta donde guardar visualizaciones
    """
    
    print("\n" + "="*80)
    print("🔥 VISUALIZADOR GRAD-CAM - Detección de Defectos")
    print("="*80)
    
    # Crear carpeta de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar modelo
    print(f"\n⏳ Cargando modelo: {Path(model_path).name}")
    classifier = EfficientNetPillClassifier()
    classifier.load_model(model_path)
    print(f"✅ Modelo cargado")
    print(f"   Umbral óptimo: {classifier.optimal_threshold:.4f}")
    
    print(f"\n📷 Procesando {len(image_paths)} imágenes...")
    print(f"📁 Guardando visualizaciones en: {output_dir}/")
    print("")
    
    # Procesar cada imagen
    results_summary = []
    
    for i, img_path in enumerate(image_paths, 1):
        img_name = Path(img_path).stem
        
        # Generar visualización Grad-CAM
        save_path = os.path.join(output_dir, f"{img_name}_gradcam.png")
        
        result = classifier.visualize_defect_gradcam(
            image_path=img_path,
            save_path=save_path,
            alpha=0.5  # 50% transparencia
        )
        
        if 'error' in result:
            print(f"❌ {i}. {Path(img_path).name}: Error - {result.get('error', 'desconocido')}")
            continue
        
        # Símbolo según clasificación
        emoji = "❌" if result['classification'] == 'NOK' else "✅"
        
        print(f"{emoji} {i}. {Path(img_path).name}")
        print(f"   Clasificación: {result['classification']}")
        print(f"   P(NOK): {result['proba_nok']:.1%}")
        print(f"   P(OK): {result['proba_ok']:.1%}")
        print(f"   💾 Guardado: {save_path}")
        print("")
        
        results_summary.append({
            'image': Path(img_path).name,
            'classification': result['classification'],
            'proba_nok': result['proba_nok'],
            'visualization': save_path
        })
    
    # Resumen
    print("=" * 80)
    print("📊 RESUMEN")
    print("=" * 80)
    
    total = len(results_summary)
    nok_count = sum(1 for r in results_summary if r['classification'] == 'NOK')
    ok_count = total - nok_count
    
    print(f"\n✅ Total procesadas: {total}")
    print(f"   OK: {ok_count} ({ok_count/total*100:.1f}%)")
    print(f"   NOK: {nok_count} ({nok_count/total*100:.1f}%)")
    print(f"\n📁 Todas las visualizaciones guardadas en: {output_dir}/")
    
    print("\n💡 Interpretación del heatmap:")
    print("   🔴 ROJO = Zonas donde el modelo detecta DEFECTOS")
    print("   🔵 AZUL = Zonas que el modelo IGNORA")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualiza dónde el modelo detecta defectos usando Grad-CAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Visualizar una imagen
  python3 visualize_gradcam.py pill.jpg
  
  # Visualizar múltiples imágenes
  python3 visualize_gradcam.py pill1.jpg pill2.jpg pill3.jpg
  
  # Visualizar todas las imágenes de una carpeta
  python3 visualize_gradcam.py images/*.jpg
  
  # Especificar modelo y carpeta de salida
  python3 visualize_gradcam.py pill.jpg --model mi_modelo.pkl --output mis_heatmaps/
        """
    )
    
    parser.add_argument('images', nargs='+', type=str,
                        help='Imágenes a procesar (una o más)')
    parser.add_argument('--model', type=str,
                        default='/models/efficientnet_pastillas.pkl',
                        help='Ruta al modelo entrenado (default: ../models/efficientnet_pastillas.pkl)')
    parser.add_argument('--output', type=str,
                        default='gradcam_visualizations',
                        help='Carpeta donde guardar visualizaciones (default: gradcam_visualizations/)')
    
    args = parser.parse_args()
    
    # Verificar que existen las imágenes
    valid_images = []
    for img_path in args.images:
        if os.path.exists(img_path):
            valid_images.append(img_path)
        else:
            print(f"⚠️  Imagen no encontrada: {img_path}")
    
    if not valid_images:
        print("\n❌ No se encontraron imágenes válidas")
        sys.exit(1)
    
    # Ejecutar visualización
    visualize_images_gradcam(
        image_paths=valid_images,
        model_path=args.model,
        output_dir=args.output
    )
