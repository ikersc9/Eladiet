from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import base64

# Importar el sistema desde el mismo directorio
from main import PillDetectionSystem


app = Flask(__name__)
CORS(app)  # Permitir peticiones desde React

# Configuración
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Configuración de modelos
DETECTOR_MODEL = "models/best-3.pt"
CLASSIFIER_MODEL = "models/autoencoder_pastillas.pkl"  # Autoencoder

# Crear directorios
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

# Inicializar sistema
try:
    pill_system = PillDetectionSystem(DETECTOR_MODEL, CLASSIFIER_MODEL)
    print("✅ Sistema de detección inicializado correctamente")
except Exception as e:
    print(f"❌ Error al inicializar el sistema: {e}")
    pill_system = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/api/health', methods=['GET'])
def health_check():
    """Verificar que el servidor está funcionando"""
    return jsonify({
        'status': 'ok',
        'system_ready': pill_system is not None,
        'classifier_available': pill_system.classifier is not None if pill_system else False
    })

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Endpoint para subir y procesar imágenes"""
    if pill_system is None:
        return jsonify({'error': 'Sistema no inicializado'}), 500
    
    if 'files' not in request.files:
        return jsonify({'error': 'No se enviaron archivos'}), 400
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No se seleccionaron archivos'}), 400
    
    uploaded_paths = []
    
    # Guardar archivos subidos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, file in enumerate(files):
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{timestamp}_{i}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            uploaded_paths.append(filepath)
    
    if not uploaded_paths:
        return jsonify({'error': 'No se pudieron procesar los archivos'}), 400
    
    try:
        # Procesar con el sistema
        metadata = pill_system.process_images(
            uploaded_paths, 
            output_base_dir=app.config['OUTPUT_FOLDER']
        )
        
        # Preparar respuesta
        response_data = {
            'success': True,
            'session_id': metadata['timestamp'],
            'total_images': len(metadata['images']),
            'results': []
        }
        
        for img_data in metadata['images']:
            # Leer imagen anotada y convertir a base64
            with open(img_data['annotated_path'], 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Calcular estadísticas
            ok_count = sum(1 for p in img_data['pills'] if p['classification'] == 'OK')
            nok_count = sum(1 for p in img_data['pills'] if p['classification'] == 'NOK')
            unknown_count = sum(1 for p in img_data['pills'] if p['classification'] == 'UNKNOWN')
            
            result = {
                'image_name': img_data['original_filename'],
                'total_pills': img_data['total_pills'],
                'annotated_image': f"data:image/jpeg;base64,{img_base64}",
                'pills': img_data['pills'],
                'stats': {
                    'ok': ok_count,
                    'nok': nok_count,
                    'unknown': unknown_count
                }
            }
            response_data['results'].append(result)
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """Listar todas las sesiones anteriores"""
    output_dir = Path(app.config['OUTPUT_FOLDER'])
    sessions = []
    
    for session_dir in sorted(output_dir.glob('session_*'), reverse=True):
        metadata_path = session_dir / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                    # Calcular estadísticas totales
                    total_pills = sum(img['total_pills'] for img in metadata['images'])
                    
                    sessions.append({
                        'id': metadata['timestamp'],
                        'timestamp': metadata['timestamp'],
                        'image_count': len(metadata['images']),
                        'total_pills': total_pills
                    })
            except Exception as e:
                print(f"Error leyendo sesión {session_dir}: {e}")
                continue
    
    return jsonify({'sessions': sessions})

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Obtener detalles de una sesión específica"""
    session_path = Path(app.config['OUTPUT_FOLDER']) / f'session_{session_id}'
    metadata_path = session_path / 'metadata.json'
    
    if not metadata_path.exists():
        return jsonify({'error': 'Sesión no encontrada'}), 404
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Añadir imágenes en base64
        for img_data in metadata['images']:
            if Path(img_data['annotated_path']).exists():
                with open(img_data['annotated_path'], 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                    img_data['annotated_image'] = f"data:image/jpeg;base64,{img_base64}"
        
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crop/<session_id>/<crop_filename>', methods=['GET'])
def get_crop(session_id, crop_filename):
    """Obtener un crop específico"""
    crop_path = Path(app.config['OUTPUT_FOLDER']) / f'session_{session_id}' / 'crops' / crop_filename
    
    if not crop_path.exists():
        return jsonify({'error': 'Crop no encontrado'}), 404
    
    return send_from_directory(crop_path.parent, crop_path.name)

@app.route('/api/config', methods=['GET'])
def get_config():
    """Obtener configuración del sistema"""
    return jsonify({
        'detector_loaded': pill_system is not None,
        'classifier_loaded': pill_system.classifier is not None if pill_system else False,
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024),
        'allowed_extensions': list(app.config['ALLOWED_EXTENSIONS'])
    })

if __name__ == '__main__':
    # Puerto configurado en 8000
    PORT = 8000
    
    print("\n" + "="*80)
    print("🔬 ELADIET - API Backend")
    print("="*80)
    print(f"🚀 Servidor iniciando en http://localhost:{PORT}")
    print(f"📁 Carpeta de uploads: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Carpeta de output: {app.config['OUTPUT_FOLDER']}")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=PORT)