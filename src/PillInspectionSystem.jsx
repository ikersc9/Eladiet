import React, { useState, useEffect, useRef } from 'react';
import { Camera, AlertTriangle, CheckCircle2, Upload, Play, Pause, TrendingUp, Activity, XCircle, Download, RefreshCw, Package, Video, X, ZoomIn } from 'lucide-react';

const PillInspectionSystem = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentImage, setCurrentImage] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [pillCrops, setPillCrops] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    ok: 0,
    defective: 0,
    byDefectType: {
      'Fracturada': 0,
      'Coloración': 0,
      'Dimensiones': 0,
      'Morfología': 0
    }
  });
  const [recentInspections, setRecentInspections] = useState([]);
  const [alert, setAlert] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [apiEndpoint, setApiEndpoint] = useState('http://localhost:8000');
  const [sessionStats, setSessionStats] = useState({ startTime: new Date(), inspections: 0 });
  const [backendStatus, setBackendStatus] = useState({ connected: false, classifier: false });
  
  // Estados para modo streaming
  const [streamMode, setStreamMode] = useState(false);
  const [imageQueue, setImageQueue] = useState([]);
  const [currentStreamIndex, setCurrentStreamIndex] = useState(0);
  const [streamInterval] = useState(5000);
  
  // ← NUEVO: Estados para zoom
  const [zoomedCrop, setZoomedCrop] = useState(null);
  
  const fileInputRef = useRef(null);
  const fileInputMultiple = useRef(null);
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch(`${apiEndpoint}/api/health`);
      if (response.ok) {
        const data = await response.json();
        setBackendStatus({
          connected: true,
          classifier: data.classifier_available
        });
      } else {
        setBackendStatus({ connected: false, classifier: false });
      }
    } catch (error) {
      setBackendStatus({ connected: false, classifier: false });
    }
  };

  const analyzeImage = async (imageFile, capturedOriginalImage = null) => {
    setProcessing(true);
    
    // Guardar la imagen original capturada para usarla en los crops
    const imageForCrops = capturedOriginalImage || originalImage;
    
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      
      const response = await fetch(`${apiEndpoint}/api/analyze`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`Error en el análisis: ${response.status}`);
      }
      
      const result = await response.json();
      processResults(result, imageForCrops);
      
      setAlert({
        type: 'success',
        message: '✅ Análisis completado correctamente con IA'
      });
      setTimeout(() => setAlert(null), 3000);
      
    } catch (error) {
      console.error('Error al analizar imagen:', error);
      setAlert({
        type: 'error',
        message: '❌ Error de conexión con el servidor. Verifica que el backend esté corriendo.'
      });
      setTimeout(() => setAlert(null), 5000);
      setProcessing(false);
    }
  };

  const processResults = (result, imageForCrops = null) => {
    const detectionsList = result.detections || [];
    
    console.log('📊 processResults llamado');
    console.log('  - imageForCrops pasado:', imageForCrops ? 'SÍ' : 'NO');
    console.log('  - Detecciones:', detectionsList.length);
    
    // Guardar la imagen del backend (con boxes)
    if (result.image_base64) {
      const imgData = `data:image/jpeg;base64,${result.image_base64}`;
      setCurrentImage(imgData);
    }
    
    const formattedDetections = detectionsList.map((det, idx) => ({
      id: idx,
      bbox: det.bbox,
      status: det.status === 'ok' ? 'ok' : 'defective',
      confidence: det.confidence * 100,
      defectType: det.defect_type || 'Desconocido'
    }));
    
    setDetections(formattedDetections);
    
    // Extraer crops usando la imagen original capturada
    setTimeout(() => {
      extractCropsFromDetections(formattedDetections, imageForCrops);
    }, 150);
    
    updateStatistics(formattedDetections);
    setProcessing(false);
  };

  const extractCropsFromDetections = (detectionsList, imageSource = null) => {
    console.log('🔍 extractCropsFromDetections llamada');
    console.log('  - Detections:', detectionsList.length);
    console.log('  - imageSource pasado:', imageSource ? 'SÍ' : 'NO');
    console.log('  - originalImage:', originalImage ? 'SÍ' : 'NO');
    console.log('  - currentImage:', currentImage ? 'SÍ' : 'NO');
    
    if (detectionsList.length === 0) {
      console.log('❌ No hay detecciones');
      return;
    }
    
    // Prioridad: imageSource > originalImage > currentImage
    const imageToUse = imageSource || originalImage || currentImage;
    if (!imageToUse) {
      console.log('❌ No hay imagen disponible');
      return;
    }
    
    console.log('✅ Usando imagen para crops');
    
    const img = new Image();
    img.src = imageToUse;
    img.onload = () => {
      console.log('✅ Imagen cargada, extrayendo crops...');
      const crops = detectionsList.map((det) => {
        const [x1, y1, x2, y2] = det.bbox;
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        const width = x2 - x1;
        const height = y2 - y1;
        canvas.width = width;
        canvas.height = height;
        
        // Extraer solo la región del crop
        ctx.drawImage(img, x1, y1, width, height, 0, 0, width, height);
        
        return {
          id: det.id,
          status: det.status,
          confidence: det.confidence,
          defectType: det.defectType,
          cropData: canvas.toDataURL('image/jpeg', 0.9)
        };
      });
      
      console.log(`✅ ${crops.length} crops extraídos exitosamente`);
      setPillCrops(crops);
    };
    img.onerror = () => {
      console.log('❌ Error al cargar imagen para crops');
    };
  };

  const handleMultipleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    
    if (files.length === 0) return;
    
    setImageQueue(files);
    setStreamMode(true);
    setCurrentStreamIndex(0);
    setIsRunning(false);
    
    setAlert({
      type: 'success',
      message: `📹 Modo streaming activado - ${files.length} imágenes en cola`
    });
    
    processNextImage(files, 0);
  };

  const processNextImage = (files, index) => {
    if (index >= files.length) {
      setStreamMode(false);
      setCurrentStreamIndex(0);
      setAlert({
        type: 'success',
        message: `✅ Stream completado - ${files.length} imágenes procesadas`
      });
      setTimeout(() => setAlert(null), 5000);
      return;
    }
    
    const file = files[index];
    const reader = new FileReader();
    
    reader.onload = (event) => {
      const imgData = event.target.result;
      // Guardar la imagen original ANTES de que se procese
      const capturedOriginalImage = imgData;
      
      setOriginalImage(imgData);
      setCurrentImage(imgData);
      
      // Pasar la imagen capturada al análisis
      analyzeImage(file, capturedOriginalImage).then(() => {
        setTimeout(() => {
          setCurrentStreamIndex(index + 1);
          processNextImage(files, index + 1);
        }, streamInterval);
      });
    };
    
    reader.readAsDataURL(file);
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const imgData = event.target.result;
        const capturedOriginalImage = imgData;
        
        setOriginalImage(imgData);
        setCurrentImage(imgData);
        setIsRunning(false);
        setStreamMode(false);
        
        analyzeImage(file, capturedOriginalImage);
      };
      reader.readAsDataURL(file);
    }
  };

  const updateStatistics = (detectionsList) => {
    let okCount = 0;
    let defectCount = 0;
    const defectTypeCount = {
      'Fracturada': 0,
      'Coloración': 0,
      'Dimensiones': 0,
      'Morfología': 0
    };

    detectionsList.forEach(det => {
      if (det.status === 'ok') {
        okCount++;
      } else {
        defectCount++;
        if (det.defectType && defectTypeCount.hasOwnProperty(det.defectType)) {
          defectTypeCount[det.defectType]++;
        }
      }
    });

    const newStats = {
      total: stats.total + detectionsList.length,
      ok: stats.ok + okCount,
      defective: stats.defective + defectCount,
      byDefectType: {
        'Fracturada': stats.byDefectType['Fracturada'] + defectTypeCount['Fracturada'],
        'Coloración': stats.byDefectType['Coloración'] + defectTypeCount['Coloración'],
        'Dimensiones': stats.byDefectType['Dimensiones'] + defectTypeCount['Dimensiones'],
        'Morfología': stats.byDefectType['Morfología'] + defectTypeCount['Morfología']
      }
    };
    
    setStats(newStats);
    setSessionStats(prev => ({ ...prev, inspections: prev.inspections + 1 }));

    const inspection = {
      timestamp: new Date().toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      pills: detectionsList.length,
      defective: defectCount,
      rate: detectionsList.length > 0 ? ((defectCount / detectionsList.length) * 100).toFixed(1) : 0
    };
    setRecentInspections(prev => [inspection, ...prev.slice(0, 9)]);

    if (detectionsList.length > 0 && (defectCount / detectionsList.length) > 0.3) {
      setAlert({
        type: 'warning',
        message: `⚠️ Tasa de rechazo elevada: ${((defectCount / detectionsList.length) * 100).toFixed(1)}%`
      });
      setTimeout(() => setAlert(null), 5000);
    }
  };

  const drawDetections = () => {
    if (!canvasRef.current || !imageRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imageRef.current;
    
    if (!img.complete) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    
    detections.forEach(det => {
      const [x1, y1, x2, y2] = det.bbox;
      const color = det.status === 'ok' ? '#059669' : '#dc2626';
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      
      const label = det.status === 'ok' ? 'CONFORME' : det.defectType.toUpperCase();
      const labelWidth = 140;
      const labelHeight = 32;
      
      ctx.fillStyle = color;
      ctx.fillRect(x1, y1 - labelHeight - 5, labelWidth, labelHeight);
      
      ctx.fillStyle = 'white';
      ctx.font = 'bold 14px Inter, system-ui, sans-serif';
      ctx.fillText(label, x1 + 8, y1 - 12);
      
      ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
      ctx.fillRect(x1, y2 + 5, 80, 24);
      ctx.fillStyle = 'white';
      ctx.font = '12px monospace';
      ctx.fillText(`${det.confidence.toFixed(1)}%`, x1 + 8, y2 + 20);
    });
  };

  useEffect(() => {
    if (currentImage && detections.length > 0) {
      drawDetections();
    }
  }, [detections, currentImage]);

  const downloadResult = () => {
    if (!canvasRef.current) return;
    const link = document.createElement('a');
    link.download = `inspeccion_eladiet_${Date.now()}.png`;
    link.href = canvasRef.current.toDataURL();
    link.click();
  };

  const resetStats = () => {
    if (window.confirm('¿Resetear todas las estadísticas?')) {
      setStats({
        total: 0,
        ok: 0,
        defective: 0,
        byDefectType: {
          'Fracturada': 0,
          'Coloración': 0,
          'Dimensiones': 0,
          'Morfología': 0
        }
      });
      setRecentInspections([]);
      setDetections([]);
      setPillCrops([]);
      setSessionStats({ startTime: new Date(), inspections: 0 });
    }
  };

  const defectRate = stats.total > 0 ? ((stats.defective / stats.total) * 100).toFixed(1) : 0;
  const currentDefectRate = detections.length > 0 
    ? ((detections.filter(d => d.status === 'defective').length / detections.length) * 100).toFixed(1) 
    : 0;

  return (
    <div className="min-h-screen bg-slate-50" style={{ fontFamily: 'Inter, system-ui, -apple-system, sans-serif' }}>
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-full mx-auto px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-8">
              <div className="flex items-center gap-4">
                <img 
                  src="/eladiet-logo.png" 
                  alt="Eladiet Logo" 
                  className="h-20 w-23 object-contain"
                />
                <div>
                  <p className="text-sm text-slate-500 uppercase tracking-widest font-semibold">
                    Control de Calidad Industrial
                  </p>
                </div>
              </div>
              
              <div className="h-12 w-px bg-slate-300"></div>
              
              <div>
                <h2 className="text-base font-semibold text-slate-700">
                  Sistema de Inspección Visual con IA
                </h2>
                <div className="flex items-center gap-4 mt-1">
                  <span className="text-sm text-slate-500">Línea A1</span>
                  <span className="text-slate-300">•</span>
                  <span className="text-sm text-slate-500">Sesión: {sessionStats.inspections} inspecciones</span>
                  <span className="text-slate-300">•</span>
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${backendStatus.connected ? 'bg-emerald-500' : 'bg-red-500'}`}></div>
                    <span className="text-xs text-slate-500">
                      {backendStatus.connected ? 'Backend conectado' : 'Backend desconectado'}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={downloadResult}
                disabled={detections.length === 0}
                className="flex items-center gap-2 px-4 py-2.5 border-2 border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition font-semibold disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Download size={18} />
                Exportar
              </button>
              
              {/* BOTÓN MODO CÁMARA */}
              <button
                onClick={() => fileInputMultiple.current.click()}
                disabled={processing || streamMode}
                className="flex items-center gap-2 px-4 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-semibold disabled:opacity-50 shadow-lg"
              >
                <Video size={18} />
                {streamMode ? 'Procesando...' : 'Modo Cámara'}
              </button>
              <input
                ref={fileInputMultiple}
                type="file"
                accept="image/*"
                multiple
                onChange={handleMultipleImageUpload}
                className="hidden"
              />
              
              <button
                onClick={() => fileInputRef.current.click()}
                disabled={processing || streamMode}
                className="flex items-center gap-2 px-4 py-2.5 border-2 border-emerald-600 text-emerald-700 rounded-lg hover:bg-emerald-50 transition font-semibold disabled:opacity-50"
              >
                <Upload size={18} />
                {processing ? 'Analizando...' : 'Cargar Imagen'}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
            </div>
          </div>
        </div>
      </header>

      {alert && (
        <div className={`border-l-4 px-8 py-3 ${
          alert.type === 'error' ? 'bg-red-50 border-red-600' : 
          alert.type === 'success' ? 'bg-emerald-50 border-emerald-600' :
          'bg-amber-50 border-amber-600'
        }`}>
          <div className="max-w-full mx-auto flex items-center gap-3">
            <AlertTriangle className={
              alert.type === 'error' ? 'text-red-700' : 
              alert.type === 'success' ? 'text-emerald-700' :
              'text-amber-700'
            } size={20} />
            <p className={`font-semibold ${
              alert.type === 'error' ? 'text-red-900' : 
              alert.type === 'success' ? 'text-emerald-900' :
              'text-amber-900'
            }`}>
              {alert.message}
            </p>
          </div>
        </div>
      )}

      <div className="max-w-full mx-auto px-8 py-6">
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-8 space-y-6">
            <div className="bg-white rounded-xl border-2 border-slate-200 shadow-sm overflow-hidden">
              <div className="px-6 py-4 border-b-2 border-slate-200 bg-slate-50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Camera className="text-slate-600" size={22} />
                    <div>
                      <span className="font-bold text-slate-900 text-base">
                        Análisis con IA
                      </span>
                      <p className="text-xs text-slate-500 mt-0.5">
                        YOLO Detection + Autoencoder Classification
                      </p>
                    </div>
                  </div>
                  
                  {streamMode && (
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-2 px-4 py-2 bg-red-50 rounded-lg border border-red-200">
                        <div className="w-2.5 h-2.5 bg-red-500 rounded-full animate-pulse"></div>
                        <span className="text-sm font-bold text-red-700">
                          STREAMING EN VIVO
                        </span>
                      </div>
                      <div className="text-sm font-semibold text-slate-700">
                        {currentStreamIndex + 1} / {imageQueue.length}
                      </div>
                    </div>
                  )}
                  
                  {processing && !streamMode && (
                    <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 rounded-lg border border-blue-200">
                      <Activity className="animate-pulse text-blue-600" size={18} />
                      <span className="text-sm font-bold text-blue-700">PROCESANDO IA...</span>
                    </div>
                  )}
                  
                  {!processing && !streamMode && backendStatus.connected && (
                    <div className="flex items-center gap-2 px-4 py-2 bg-emerald-50 rounded-lg border border-emerald-200">
                      <div className="w-2.5 h-2.5 bg-emerald-500 rounded-full"></div>
                      <span className="text-sm font-bold text-emerald-700">SISTEMA LISTO</span>
                    </div>
                  )}
                </div>
                
                {streamMode && (
                  <div className="mt-3">
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div 
                        className="bg-red-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${((currentStreamIndex + 1) / imageQueue.length) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>

              <div className="relative bg-slate-900" style={{ height: '500px' }}>
                {currentImage ? (
                  <div className="relative w-full h-full">
                    <img 
                      ref={imageRef}
                      src={currentImage} 
                      alt="Inspection" 
                      className="absolute inset-0 w-full h-full object-contain"
                      onLoad={drawDetections}
                    />
                    {detections.length > 0 && (
                      <canvas
                        ref={canvasRef}
                        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                      />
                    )}
                  </div>
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <div className="text-center">
                      <Camera className="text-slate-600 mx-auto mb-4" size={64} strokeWidth={1.5} />
                      <p className="text-slate-400 text-lg font-semibold mb-2">
                        Sistema en espera
                      </p>
                      <p className="text-slate-500 text-sm">
                        Cargar imagen o activar modo cámara
                      </p>
                      {!backendStatus.connected && (
                        <p className="text-red-400 text-sm mt-4">
                          ⚠️ Backend no conectado
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {detections.length > 0 && (
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-white border-2 border-slate-200 rounded-xl p-5 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-bold text-slate-600 uppercase tracking-wide">Total</span>
                    <Package className="text-slate-400" size={20} />
                  </div>
                  <p className="text-4xl font-bold text-slate-900">
                    {detections.length}
                  </p>
                  <p className="text-xs text-slate-500 mt-1">Unidades detectadas</p>
                </div>
                
                <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 border-2 border-emerald-300 rounded-xl p-5 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-bold text-emerald-800 uppercase tracking-wide">Conformes</span>
                    <CheckCircle2 className="text-emerald-600" size={20} />
                  </div>
                  <p className="text-4xl font-bold text-emerald-700">
                    {detections.filter(d => d.status === 'ok').length}
                  </p>
                  <p className="text-xs text-emerald-700 mt-1 font-semibold">
                    {((detections.filter(d => d.status === 'ok').length / detections.length) * 100).toFixed(1)}% aprobadas
                  </p>
                </div>
                
                <div className="bg-gradient-to-br from-red-50 to-red-100 border-2 border-red-300 rounded-xl p-5 shadow-sm">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-bold text-red-800 uppercase tracking-wide">Rechazadas</span>
                    <XCircle className="text-red-600" size={20} />
                  </div>
                  <p className="text-4xl font-bold text-red-700">
                    {detections.filter(d => d.status === 'defective').length}
                  </p>
                  <p className="text-xs text-red-700 mt-1 font-semibold">
                    {currentDefectRate}% defectuosas
                  </p>
                </div>
              </div>
            )}
            
            {/* GALERÍA DE CROPS CON ZOOM */}
            {pillCrops.length > 0 && (
              <div className="bg-white rounded-xl border-2 border-slate-200 shadow-sm p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-bold text-slate-900 text-base flex items-center gap-2">
                    <Package size={20} />
                    Pastillas Detectadas ({pillCrops.length})
                  </h3>
                  <span className="text-xs text-slate-500 flex items-center gap-1">
                    <ZoomIn size={14} />
                    Click para ampliar • Desliza →
                  </span>
                </div>
                
                <div className="flex gap-4 overflow-x-auto pb-4" style={{ scrollbarWidth: 'thin' }}>
                  {pillCrops.map((crop) => (
                    <div 
                      key={crop.id}
                      onClick={() => setZoomedCrop(crop)}
                      className="flex-shrink-0 w-36 h-36 relative rounded-lg overflow-hidden border-4 shadow-md hover:shadow-2xl transition-all cursor-pointer hover:scale-105"
                      style={{
                        borderColor: crop.status === 'ok' ? '#059669' : '#dc2626'
                      }}
                    >
                      <img 
                        src={crop.cropData} 
                        alt={`Pastilla ${crop.id + 1}`}
                        className="w-full h-full object-cover"
                      />
                      
                      <div className={`absolute bottom-0 left-0 right-0 py-1.5 text-center text-sm font-bold text-white ${
                        crop.status === 'ok' ? 'bg-emerald-600' : 'bg-red-600'
                      }`}>
                        #{crop.id + 1} {crop.status === 'ok' ? '✅' : '❌'}
                      </div>
                      
                      <div className="absolute top-2 right-2 bg-black/75 text-white text-xs px-2 py-1 rounded font-mono">
                        {crop.confidence.toFixed(0)}%
                      </div>
                      
                      {crop.status === 'defective' && (
                        <div className="absolute top-2 left-2 bg-red-600 text-white text-xs px-2 py-1 rounded font-semibold">
                          {crop.defectType}
                        </div>
                      )}
                      
                      {/* Icono de zoom */}
                      <div className="absolute inset-0 bg-black/0 hover:bg-black/20 transition-all flex items-center justify-center">
                        <ZoomIn className="text-white opacity-0 hover:opacity-100 transition-opacity" size={32} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="col-span-4 space-y-6">
            <div className="bg-white rounded-xl border-2 border-slate-200 shadow-sm p-6">
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2">
                  <TrendingUp className="text-slate-600" size={20} />
                  <h3 className="font-bold text-slate-900 text-base">Métricas Acumuladas</h3>
                </div>
                <button
                  onClick={resetStats}
                  className="text-slate-400 hover:text-slate-600 transition"
                  title="Resetear"
                >
                  <RefreshCw size={16} />
                </button>
              </div>

              <div className="space-y-5">
                <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs font-bold text-slate-600 uppercase tracking-wider">Total</span>
                    <span className="text-2xl font-bold text-slate-900">{stats.total}</span>
                  </div>
                  <div className="w-full bg-slate-300 rounded-full h-2">
                    <div className="bg-slate-700 h-2 rounded-full" style={{ width: '100%' }}></div>
                  </div>
                </div>

                <div className="p-4 bg-emerald-50 rounded-lg border border-emerald-200">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs font-bold text-emerald-700 uppercase tracking-wider">Conformes</span>
                    <span className="text-2xl font-bold text-emerald-700">{stats.ok}</span>
                  </div>
                  <div className="w-full bg-emerald-200 rounded-full h-2">
                    <div 
                      className="bg-emerald-600 h-2 rounded-full transition-all duration-500" 
                      style={{ width: stats.total > 0 ? `${(stats.ok / stats.total) * 100}%` : '0%' }}
                    ></div>
                  </div>
                </div>

                <div className="p-4 bg-red-50 rounded-lg border border-red-200">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs font-bold text-red-700 uppercase tracking-wider">Rechazadas</span>
                    <span className="text-2xl font-bold text-red-700">{stats.defective}</span>
                  </div>
                  <div className="w-full bg-red-200 rounded-full h-2">
                    <div 
                      className="bg-red-600 h-2 rounded-full transition-all duration-500" 
                      style={{ width: stats.total > 0 ? `${(stats.defective / stats.total) * 100}%` : '0%' }}
                    ></div>
                  </div>
                </div>

                <div className="pt-4 border-t-2 border-slate-200">
                  <p className="text-xs text-slate-500 uppercase tracking-widest font-bold mb-3 text-center">Tasa Global de Rechazo</p>
                  <p className={`text-6xl font-bold text-center ${defectRate > 20 ? 'text-red-600' : 'text-emerald-600'}`}>
                    {defectRate}<span className="text-3xl">%</span>
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl border-2 border-slate-200 shadow-sm p-6">
              <h3 className="font-bold text-slate-900 mb-5 text-base">Clasificación de Defectos</h3>
              <div className="space-y-4">
                {Object.entries(stats.byDefectType).map(([type, count]) => (
                  <div key={type}>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-slate-700 font-semibold">{type}</span>
                      <span className="font-bold text-slate-900">{count}</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div 
                        className="bg-orange-500 h-2 rounded-full transition-all duration-500" 
                        style={{ width: stats.defective > 0 ? `${(count / stats.defective) * 100}%` : '0%' }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-xl border-2 border-slate-200 shadow-sm p-6">
              <h3 className="font-bold text-slate-900 mb-4 text-base">Registro de Inspecciones</h3>
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {recentInspections.length === 0 ? (
                  <p className="text-sm text-slate-500 text-center py-6">Sin datos disponibles</p>
                ) : (
                  recentInspections.map((inspection, idx) => (
                    <div key={idx} className="flex items-center justify-between text-sm p-3 bg-slate-50 rounded-lg border border-slate-200">
                      <span className="text-slate-600 font-mono text-xs">{inspection.timestamp}</span>
                      <span className="font-bold text-slate-700">
                        {inspection.defective}/{inspection.pills}
                      </span>
                      <span className={`font-bold px-2 py-1 rounded ${
                        parseFloat(inspection.rate) > 20 
                          ? 'bg-red-100 text-red-700' 
                          : 'bg-emerald-100 text-emerald-700'
                      }`}>
                        {inspection.rate}%
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* MODAL DE ZOOM */}
      {zoomedCrop && (
        <div 
          className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-8"
          onClick={() => setZoomedCrop(null)}
        >
          <div className="relative max-w-4xl max-h-full" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setZoomedCrop(null)}
              className="absolute -top-12 right-0 text-white hover:text-red-500 transition"
            >
              <X size={32} />
            </button>
            
            <div className="bg-white rounded-xl p-6 shadow-2xl">
              <div className="mb-4">
                <h3 className="text-2xl font-bold text-slate-900">
                  Pastilla #{zoomedCrop.id + 1}
                </h3>
                <div className="flex items-center gap-4 mt-2">
                  <span className={`px-4 py-2 rounded-lg font-bold text-lg ${
                    zoomedCrop.status === 'ok' 
                      ? 'bg-emerald-100 text-emerald-700' 
                      : 'bg-red-100 text-red-700'
                  }`}>
                    {zoomedCrop.status === 'ok' ? '✅ CONFORME' : '❌ RECHAZADA'}
                  </span>
                  <span className="text-slate-600">
                    Confianza: <span className="font-bold">{zoomedCrop.confidence.toFixed(1)}%</span>
                  </span>
                  {zoomedCrop.status === 'defective' && (
                    <span className="px-3 py-1 bg-red-600 text-white rounded font-semibold">
                      {zoomedCrop.defectType}
                    </span>
                  )}
                </div>
              </div>
              
              <img 
                src={zoomedCrop.cropData} 
                alt={`Pastilla ${zoomedCrop.id + 1} ampliada`}
                className="w-full h-auto rounded-lg border-4"
                style={{
                  borderColor: zoomedCrop.status === 'ok' ? '#059669' : '#dc2626',
                  maxHeight: '70vh',
                  objectFit: 'contain'
                }}
              />
              
              <p className="text-center text-slate-500 text-sm mt-4">
                Click fuera de la imagen o en la X para cerrar
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PillInspectionSystem;