/**
 * PROXIMA KWS LAB — Developer Testing Panel Client Logic
 * Handles real-time Web Audio API stream, audio file upload, microphone recording,
 * dynamic TFLite model swapping, live detection indicators, and architecture visualization.
 */

// Universal high-quality linear audio resampler
function resampleBuffer(buffer, fromSampleRate, toSampleRate = 16000) {
  if (!buffer || buffer.length === 0 || fromSampleRate === toSampleRate) {
    return buffer;
  }
  const ratio = fromSampleRate / toSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const origIdx = i * ratio;
    const index1 = Math.floor(origIdx);
    const index2 = Math.min(index1 + 1, buffer.length - 1);
    const frac = origIdx - index1;
    result[i] = buffer[index1] * (1 - frac) + buffer[index2] * frac;
  }
  return result;
}

class ProximaKWSClient {
  constructor() {
    // State
    this.isLiveListening = false;
    this.isRecording = false;
    this.audioContext = null;
    this.micStream = null;
    this.liveSource = null;
    this.muteGain = null;
    this.scriptProcessor = null;
    this.ringBuffer = new Float32Array(32000); // exactly 2.0s at 16kHz
    this.ringBufferIndex = 0;
    
    // Recording state (MediaRecorder based)
    this.mediaRecorder = null;
    this.recordedChunks = [];
    this.recordedBlob = null;
    this.recordAudioContext = null;
    this.recordSource = null;
    this.recordAnalyser = null;
    this.recordingTimer = null;
    
    this.selectedUploadFile = null;
    this.lastDetectionTime = 0;
    this.lastLiveDetectionTime = 0;
    this.liveCooldownMs = 2500; // 2.5s refractory cooldown (exceeds 2.0s buffer to guarantee 1 utterance = 1 count)
    this.liveDetectedState = false;
    this.isStreamBusy = false; // Prevent overlapping asynchronous requests
    this.detectionHoldMs = 1200; // Hold green indicator for visibility
    this.detectionTimer = null;
    
    // Noise Rejection & Gating Configuration
    this.energyGateEnabled = true;
    this.energyThresholdDb = -42.0;
    this.temporalSmoothingEnabled = true;
    this.consecutiveProximaHits = 0;
    
    // Metrics
    this.totalInferences = 0;
    this.proximaHits = 0;
    this.unknownHits = 0;
    this.historyRecords = [];
    this.threshold = 60; // 60% default confidence threshold

    // DOM Elements
    this.initDOMElements();
    this.bindEvents();
    
    // Initial Load
    this.fetchModels();
    this.fetchTestSamples();
    this.initWaveformCanvas();
  }

  initDOMElements() {
    this.modelSelect = document.getElementById('modelSelect');
    this.refreshModelsBtn = document.getElementById('refreshModelsBtn');
    this.detectionCard = document.getElementById('detectionCard');
    this.detectionTitle = document.getElementById('detectionTitle');
    this.detectionSubtitle = document.getElementById('detectionSubtitle');
    this.listeningStatusPill = document.getElementById('listeningStatusPill');
    this.listeningStatusText = document.getElementById('listeningStatusText');
    this.currentModelDisplay = document.getElementById('currentModelDisplay');
    
    this.confProximaVal = document.getElementById('confProximaVal');
    this.confUnknownVal = document.getElementById('confUnknownVal');
    this.confBarFill = document.getElementById('confBarFill');
    
    this.metricLatency = document.getElementById('metricLatency');
    this.metricPreproc = document.getElementById('metricPreproc');
    this.metricHits = document.getElementById('metricHits');
    this.metricTotalTests = document.getElementById('metricTotalTests');
    this.metricFlashBudget = document.getElementById('metricFlashBudget');
    this.budgetProgressFill = document.getElementById('budgetProgressFill');
    this.metricCpuVal = document.getElementById('metricCpuVal');
    this.cpuProgressFill = document.getElementById('cpuProgressFill');
    this.metricCpuSub = document.getElementById('metricCpuSub');
    this.thresholdSlider = document.getElementById('thresholdSlider');
    this.thresholdVal = document.getElementById('thresholdVal');
    
    // Mode 1: Live Voice Controls
    this.toggleLiveBtn = document.getElementById('toggleLiveBtn');
    this.liveBtnText = document.getElementById('liveBtnText');
    this.micLevelContainer = document.getElementById('micLevelContainer');
    this.micLevelBar = document.getElementById('micLevelBar');
    this.micDbLabel = document.getElementById('micDbLabel');
    this.micErrorBanner = document.getElementById('micErrorBanner');
    this.micErrorDesc = document.getElementById('micErrorDesc');
    
    // Noise Rejection & Gating Controls
    this.toggleEnergyGate = document.getElementById('toggleEnergyGate');
    this.gateThresholdSlider = document.getElementById('gateThresholdSlider');
    this.gateThresholdVal = document.getElementById('gateThresholdVal');
    this.gateStatusText = document.getElementById('gateStatusText');
    this.gateSliderRow = document.getElementById('gateSliderRow');
    this.toggleSmoothing = document.getElementById('toggleSmoothing');
    this.smoothingStatusText = document.getElementById('smoothingStatusText');
    this.cooldownSlider = document.getElementById('cooldownSlider');
    this.cooldownStatusText = document.getElementById('cooldownStatusText');
    this.cooldownValDisplay = document.getElementById('cooldownValDisplay');
    
    // Mode 2: Audio File Upload Controls
    this.fileDropzone = document.getElementById('fileDropzone');
    this.audioFileInput = document.getElementById('audioFileInput');
    this.uploadPreviewContainer = document.getElementById('uploadPreviewContainer');
    this.uploadFileName = document.getElementById('uploadFileName');
    this.uploadFileMeta = document.getElementById('uploadFileMeta');
    this.uploadedAudioPlayer = document.getElementById('uploadedAudioPlayer');
    this.testUploadBtn = document.getElementById('testUploadBtn');

    // Mode 3: Recording Controls
    this.startRecordBtn = document.getElementById('startRecordBtn');
    this.stopRecordBtn = document.getElementById('stopRecordBtn');
    this.testRecordBtn = document.getElementById('testRecordBtn');
    this.recordingTimerBadge = document.getElementById('recordingTimerBadge');
    this.recordLevelContainer = document.getElementById('recordLevelContainer');
    this.recordLevelBar = document.getElementById('recordLevelBar');
    this.recordDbLabel = document.getElementById('recordDbLabel');
    this.playbackContainer = document.getElementById('playbackContainer');
    this.recordedAudioPlayer = document.getElementById('recordedAudioPlayer');
    
    // Mode 4: Test Sample Elements
    this.sampleSelect = document.getElementById('sampleSelect');
    this.testSampleBtn = document.getElementById('testSampleBtn');
    
    // Info Elements
    this.infoName = document.getElementById('infoName');
    this.infoInput = document.getElementById('infoInput');
    this.infoQuant = document.getElementById('infoQuant');
    this.infoParams = document.getElementById('infoParams');
    this.infoSize = document.getElementById('infoSize');
    this.infoArena = document.getElementById('infoArena');
    this.infoBudget = document.getElementById('infoBudget');
    this.quantBadge = document.getElementById('quantBadge');
    
    // Pipeline & History Elements
    this.pipelineFlowContainer = document.getElementById('pipelineFlowContainer');
    this.historyTableBody = document.getElementById('historyTableBody');
    this.clearHistoryBtn = document.getElementById('clearHistoryBtn');
    this.historyCountBadge = document.getElementById('historyCountBadge');
    this.waveformCanvas = document.getElementById('waveformCanvas');
    this.vizMetrics = document.getElementById('vizMetrics');
  }

  bindEvents() {
    this.modelSelect.addEventListener('change', (e) => this.switchModel(e.target.value));
    this.modelSelect.addEventListener('input', (e) => this.switchModel(e.target.value));
    this.refreshModelsBtn.addEventListener('click', () => this.fetchModels());
    
    // Live Listening
    this.toggleLiveBtn.addEventListener('click', () => this.toggleLiveListening());
    
    // File Upload
    this.fileDropzone.addEventListener('click', () => this.audioFileInput.click());
    this.audioFileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));
    
    this.fileDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.fileDropzone.classList.add('dragover');
    });
    this.fileDropzone.addEventListener('dragleave', () => {
      this.fileDropzone.classList.remove('dragover');
    });
    this.fileDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.fileDropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        this.handleFileSelect(e.dataTransfer.files[0]);
      }
    });
    this.testUploadBtn.addEventListener('click', () => this.testUploadedAudio());

    // Recording
    this.startRecordBtn.addEventListener('click', () => this.startRecording());
    this.stopRecordBtn.addEventListener('click', () => this.stopRecording());
    this.testRecordBtn.addEventListener('click', () => this.testRecordedAudio());
    
    // Test Samples & History
    this.testSampleBtn.addEventListener('click', () => this.testDatasetSample());
    this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
    
    // Sensitivity Threshold Slider
    this.thresholdSlider.addEventListener('input', (e) => {
      this.threshold = parseInt(e.target.value, 10);
      this.thresholdVal.textContent = `${this.threshold}%`;
    });

    // Reset Hits / Inferences Counter
    const resetHitsBtn = document.getElementById('resetHitsBtn');
    if (resetHitsBtn) {
      resetHitsBtn.addEventListener('click', () => {
        this.proximaHits = 0;
        this.unknownHits = 0;
        this.totalInferences = 0;
        this.metricHits.textContent = '0';
        this.metricTotalTests.textContent = '0 total inferences';
      });
    }

    // RMS Energy Gate Toggle
    if (this.toggleEnergyGate) {
      this.toggleEnergyGate.addEventListener('change', (e) => {
        this.energyGateEnabled = e.target.checked;
        if (this.gateSliderRow) this.gateSliderRow.style.display = this.energyGateEnabled ? 'flex' : 'none';
        if (this.gateStatusText) {
          this.gateStatusText.innerHTML = this.energyGateEnabled 
            ? `Active (<b id="gateThresholdVal">${this.energyThresholdDb} dB</b> cutoff)` 
            : '<span style="color: #94a3b8;">Disabled (raw microphone stream)</span>';
        }
      });
    }

    // RMS Energy Gate Slider
    if (this.gateThresholdSlider) {
      this.gateThresholdSlider.addEventListener('input', (e) => {
        this.energyThresholdDb = parseFloat(e.target.value);
        const valElem = document.getElementById('gateThresholdVal');
        if (valElem) valElem.textContent = `${this.energyThresholdDb} dB`;
      });
    }

    // Temporal Smoothing Toggle
    if (this.toggleSmoothing) {
      this.toggleSmoothing.addEventListener('change', (e) => {
        this.temporalSmoothingEnabled = e.target.checked;
        this.consecutiveProximaHits = 0;
        if (this.smoothingStatusText) {
          this.smoothingStatusText.textContent = this.temporalSmoothingEnabled
            ? 'Active (discards transient 1-frame glitches)'
            : 'Disabled (instant 1-frame trigger)';
        }
      });
    }

    // Refractory Cooldown Slider
    if (this.cooldownSlider) {
      this.cooldownSlider.addEventListener('input', (e) => {
        this.liveCooldownMs = parseInt(e.target.value, 10);
        const secText = `${(this.liveCooldownMs / 1000).toFixed(1)}s`;
        if (this.cooldownValDisplay) {
          this.cooldownValDisplay.textContent = secText;
        }
        if (this.cooldownStatusText) {
          this.cooldownStatusText.innerHTML = `Active (<b id="cooldownValDisplay">${secText}</b> lockout window)`;
        }
      });
    }
  }

  /* ========================================================================
     MODEL MANAGEMENT (CORE FEATURE 3 & 4)
     ======================================================================== */
  async fetchModels() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      
      this.modelSelect.innerHTML = '';
      data.models.forEach((m) => {
        const opt = document.createElement('option');
        opt.value = m.filename;
        opt.textContent = `${m.filename} (${m.size_kb} KB, ${m.quantization})`;
        if (m.filename === data.active_model) {
          opt.selected = true;
        }
        this.modelSelect.appendChild(opt);
      });
      
      this.modelSelect.value = data.active_model;
      if (this.currentModelDisplay) {
        this.currentModelDisplay.textContent = data.active_model;
      }
      await this.fetchModelInfo();
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  }

  async switchModel(modelName) {
    if (!modelName) return;
    try {
      const res = await fetch('/api/models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName })
      });
      const data = await res.json();
      if (data.status === 'success') {
        this.modelSelect.value = data.active_model;
        if (this.currentModelDisplay) {
          this.currentModelDisplay.textContent = data.active_model;
        }
        this.renderModelInfo(data.model_info);
        this.renderArchitecture(data.architecture);
      }
    } catch (err) {
      console.error('Failed to switch model:', err);
      alert('Error switching model: ' + err.message);
    }
  }

  async fetchModelInfo() {
    try {
      const res = await fetch('/api/model/info');
      const data = await res.json();
      this.renderModelInfo(data.info);
      this.renderArchitecture(data.architecture);
    } catch (err) {
      console.error('Failed to fetch model info:', err);
    }
  }

  renderModelInfo(info) {
    if (!info) return;
    this.infoName.textContent = info.filename;
    this.infoInput.textContent = info.input_shape_str || '201 × 40 × 1';
    this.infoQuant.textContent = info.quantization;
    this.infoParams.textContent = info.parameter_count.toLocaleString();
    this.infoSize.textContent = `${info.size_kb} KB (${info.size_bytes.toLocaleString()} B)`;
    this.infoArena.textContent = `~${info.tensor_arena_estimate_kb} KB RAM`;
    this.infoBudget.textContent = `${info.budget_pct}% of 256 KB`;
    
    this.metricFlashBudget.textContent = `${info.budget_pct}%`;
    this.budgetProgressFill.style.width = `${Math.min(info.budget_pct, 100)}%`;
    
    this.quantBadge.textContent = info.quantization;
    if (info.quantization.includes('INT8')) {
      this.quantBadge.className = 'badge badge-success';
    } else {
      this.quantBadge.className = 'badge';
    }
  }

  /* ========================================================================
     NEURAL NETWORK ARCHITECTURE VISUALIZATION (CORE FEATURE 5)
     ======================================================================== */
  renderArchitecture(nodes) {
    if (!nodes || !nodes.length) return;
    this.pipelineFlowContainer.innerHTML = '';
    
    nodes.forEach((node, idx) => {
      const card = document.createElement('div');
      card.className = 'arch-node';
      card.innerHTML = `
        <div class="arch-node-title">${node.label}</div>
        <div class="arch-node-badge">${node.badge}</div>
        <div class="arch-node-details">${node.details}</div>
      `;
      this.pipelineFlowContainer.appendChild(card);
      
      if (idx < nodes.length - 1) {
        const connector = document.createElement('div');
        connector.className = 'arch-connector';
        connector.textContent = '→';
        this.pipelineFlowContainer.appendChild(connector);
      }
    });
  }

  /* ========================================================================
     LIVE VOICE LISTENING MODE (CORE FEATURE 1)
     ======================================================================== */
  async toggleLiveListening() {
    if (this.isLiveListening) {
      this.stopLiveListening();
    } else {
      await this.startLiveListening();
    }
  }

  async startLiveListening() {
    if (this.micErrorBanner) this.micErrorBanner.style.display = 'none';

    try {
      // Unconstrained audio capture for full hardware compatibility
      this.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      const nativeSr = this.audioContext.sampleRate;
      console.log(`[Live Mic] AudioContext native rate: ${nativeSr} Hz`);

      // Store source on instance so garbage collection cannot detach it
      this.liveSource = this.audioContext.createMediaStreamSource(this.micStream);
      this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
      
      // Mute gain node avoids browser acoustic echo suppression from zeroing input!
      this.muteGain = this.audioContext.createGain();
      this.muteGain.gain.value = 0.0;

      this.ringBuffer = new Float32Array(32000); // exactly 2.0s at 16,000 Hz
      this.ringBufferIndex = 0;
      let lastSendTime = 0;
      let bufferCount = 0;

      this.scriptProcessor.onaudioprocess = (e) => {
        if (!this.isLiveListening) return;

        const inputData = e.inputBuffer.getChannelData(0);
        bufferCount++;

        // 1. Resample incoming chunk from native rate to 16,000 Hz
        const resampled16k = resampleBuffer(inputData, nativeSr, 16000);
        
        // 2. Push resampled samples into circular 2-second buffer
        for (let i = 0; i < resampled16k.length; i++) {
          this.ringBuffer[this.ringBufferIndex] = resampled16k[i];
          this.ringBufferIndex = (this.ringBufferIndex + 1) % 32000;
        }

        // 3. Audio Activity / Volume Meter
        let sumSq = 0;
        for (let i = 0; i < inputData.length; i++) {
          sumSq += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sumSq / inputData.length);
        const db = 20 * Math.log10(Math.max(rms, 1e-4));
        const pct = Math.min(Math.max((db + 50) * 2, 0), 100);
        if (this.micLevelBar) this.micLevelBar.style.width = `${pct}%`;
        if (this.micDbLabel) this.micDbLabel.textContent = `${db.toFixed(1)} dB`;

        // 4. Draw real-time oscilloscope
        this.drawWaveform(inputData);

        // 5. Send latest 2s window for inference every ~250 ms
        const now = Date.now();
        if (now - lastSendTime > 250) {
          lastSendTime = now;
          this.sendStreamInference();
        }

        if (this.vizMetrics) {
          this.vizMetrics.textContent = `Rate: ${nativeSr}Hz → 16kHz | Buffers: ${bufferCount}`;
        }
      };

      // Connect: liveSource -> scriptProcessor -> muteGain -> destination
      this.liveSource.connect(this.scriptProcessor);
      this.scriptProcessor.connect(this.muteGain);
      this.muteGain.connect(this.audioContext.destination);

      this.isLiveListening = true;
      this.toggleLiveBtn.classList.add('active');
      this.liveBtnText.textContent = 'Stop Live Listening';
      if (this.micLevelContainer) this.micLevelContainer.style.display = 'block';
      this.setListeningState('LISTENING');
    } catch (err) {
      console.error('Microphone error:', err);
      if (this.micErrorBanner) {
        this.micErrorBanner.style.display = 'block';
        this.micErrorDesc.textContent = `Microphone access failed: ${err.message}. Please allow microphone permissions in your browser URL bar.`;
      }
      this.stopLiveListening();
    }
  }

  stopLiveListening() {
    this.isLiveListening = false;
    this.isStreamBusy = false;
    this.liveDetectedState = false;
    this.lastLiveDetectionTime = 0;
    this.consecutiveProximaHits = 0;
    this.ringBuffer.fill(0);
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }
    if (this.liveSource) {
      this.liveSource.disconnect();
      this.liveSource = null;
    }
    if (this.muteGain) {
      this.muteGain.disconnect();
      this.muteGain = null;
    }
    if (this.micStream) {
      this.micStream.getTracks().forEach((t) => t.stop());
      this.micStream = null;
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
    
    this.toggleLiveBtn.classList.remove('active');
    this.liveBtnText.textContent = 'Start Live Listening';
    if (this.micLevelContainer) this.micLevelContainer.style.display = 'none';
    this.setListeningState('STANDBY');
    this.clearWaveform();
  }

  async sendStreamInference() {
    if (!this.isLiveListening) return;
    if (this.isStreamBusy) return; // Prevent concurrent requests from colliding and causing out-of-order triggers

    this.isStreamBusy = true;
    try {
      // Linearize circular buffer into continuous 2.0s 16kHz array
      const orderedSamples = new Float32Array(32000);
      const startIdx = this.ringBufferIndex;
      const len1 = 32000 - startIdx;
      orderedSamples.set(this.ringBuffer.subarray(startIdx), 0);
      orderedSamples.set(this.ringBuffer.subarray(0, startIdx), len1);

      const res = await fetch('/api/predict/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          samples: Array.from(orderedSamples),
          sample_rate: 16000,
          enable_energy_gate: this.energyGateEnabled,
          energy_threshold_db: this.energyThresholdDb,
          model_name: this.modelSelect ? this.modelSelect.value : undefined
        })
      });
      if (res.ok) {
        const data = await res.json();
        this.handlePredictionResult(data, 'Live Stream');
      }
    } catch (err) {
      console.error('Live stream inference error:', err);
    } finally {
      this.isStreamBusy = false;
    }
  }

  /* ========================================================================
     AUDIO FILE UPLOAD & TESTING (MODE 2)
     ======================================================================== */
  handleFileSelect(file) {
    if (!file) return;
    this.selectedUploadFile = file;
    this.uploadFileName.textContent = file.name;
    const sizeKb = (file.size / 1024).toFixed(1);
    this.uploadFileMeta.textContent = `${sizeKb} KB • ${file.type || 'audio'}`;
    
    // Set preview player
    const objectUrl = URL.createObjectURL(file);
    this.uploadedAudioPlayer.src = objectUrl;
    this.uploadPreviewContainer.style.display = 'block';
  }

  async testUploadedAudio() {
    if (!this.selectedUploadFile) return;

    this.testUploadBtn.disabled = true;
    this.testUploadBtn.textContent = 'Processing Audio...';

    try {
      const formData = new FormData();
      formData.append('file', this.selectedUploadFile, this.selectedUploadFile.name);
      if (this.modelSelect && this.modelSelect.value) {
        formData.append('model_name', this.modelSelect.value);
      }

      const res = await fetch('/api/predict/audio', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `Server error: ${res.status}`);
      }

      this.handlePredictionResult(data, `Uploaded File (${this.selectedUploadFile.name})`);
    } catch (err) {
      console.error('Upload test error:', err);
      alert('Upload test failed: ' + err.message);
    } finally {
      this.testUploadBtn.disabled = false;
      this.testUploadBtn.textContent = '⚡ Test Uploaded Audio';
    }
  }

  /* ========================================================================
     AUDIO RECORDING / TESTING (MODE 3 — Robust Native MediaRecorder)
     ======================================================================== */
  async startRecording() {
    try {
      this.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // 1. AudioContext + Analyser purely for real-time visual level meter & oscilloscope
      this.recordAudioContext = new (window.AudioContext || window.webkitAudioContext)();
      if (this.recordAudioContext.state === 'suspended') {
        await this.recordAudioContext.resume();
      }
      this.recordSource = this.recordAudioContext.createMediaStreamSource(this.micStream);
      this.recordAnalyser = this.recordAudioContext.createAnalyser();
      this.recordAnalyser.fftSize = 512;
      this.recordSource.connect(this.recordAnalyser);

      // Metering & visualizer animation loop
      const pcmBuffer = new Float32Array(this.recordAnalyser.fftSize);
      const updateVisuals = () => {
        if (!this.isRecording) return;
        this.recordAnalyser.getFloatTimeDomainData(pcmBuffer);
        this.drawWaveform(pcmBuffer);
        
        let sumSq = 0;
        for (let i = 0; i < pcmBuffer.length; i++) sumSq += pcmBuffer[i] * pcmBuffer[i];
        const rms = Math.sqrt(sumSq / pcmBuffer.length);
        const db = 20 * Math.log10(Math.max(rms, 1e-4));
        const pct = Math.min(Math.max((db + 50) * 2, 0), 100);
        if (this.recordLevelBar) this.recordLevelBar.style.width = `${pct}%`;
        if (this.recordDbLabel) this.recordDbLabel.textContent = `${db.toFixed(1)} dB`;
        
        requestAnimationFrame(updateVisuals);
      };
      requestAnimationFrame(updateVisuals);

      // 2. Hardware-level native recording using MediaRecorder
      this.recordedChunks = [];
      let mimeType = '';
      if (typeof MediaRecorder.isTypeSupported === 'function') {
        if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
          mimeType = 'audio/webm;codecs=opus';
        } else if (MediaRecorder.isTypeSupported('audio/webm')) {
          mimeType = 'audio/webm';
        } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
          mimeType = 'audio/ogg;codecs=opus';
        }
      }
      
      this.mediaRecorder = mimeType ? new MediaRecorder(this.micStream, { mimeType }) : new MediaRecorder(this.micStream);
      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          this.recordedChunks.push(e.data);
        }
      };

      this.isRecording = true;
      this.startRecordBtn.disabled = true;
      this.stopRecordBtn.disabled = false;
      this.testRecordBtn.disabled = true;
      if (this.recordLevelContainer) this.recordLevelContainer.style.display = 'block';
      this.setListeningState('RECORDING');

      const startTime = Date.now();
      this.recordingTimer = setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000;
        this.recordingTimerBadge.textContent = `${elapsed.toFixed(1)}s / 2.0s`;
        if (elapsed >= 2.5) {
          this.stopRecording();
        }
      }, 100);

      this.mediaRecorder.start(100); // Deliver chunks every 100ms
    } catch (err) {
      console.error('Failed to start recording:', err);
      alert('Microphone recording error: ' + err.message);
      this.stopRecording();
    }
  }

  async stopRecording() {
    if (!this.isRecording) return;
    this.isRecording = false;

    if (this.recordingTimer) {
      clearInterval(this.recordingTimer);
      this.recordingTimer = null;
    }

    if (this.recordLevelContainer) this.recordLevelContainer.style.display = 'none';
    this.startRecordBtn.disabled = false;
    this.stopRecordBtn.disabled = true;
    this.setListeningState('STANDBY');

    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }

    // Wait for the final audio chunk to be delivered
    await new Promise((r) => setTimeout(r, 150));

    if (this.micStream) {
      this.micStream.getTracks().forEach((t) => t.stop());
      this.micStream = null;
    }
    if (this.recordAudioContext && this.recordAudioContext.state !== 'closed') {
      this.recordAudioContext.close().catch(() => {});
      this.recordAudioContext = null;
    }

    // Decode native audio chunks into standard 16 kHz WAV
    if (this.recordedChunks.length > 0) {
      try {
        const rawBlob = new Blob(this.recordedChunks, { type: this.mediaRecorder.mimeType || 'audio/webm' });
        const arrayBuffer = await rawBlob.arrayBuffer();
        const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
        const decodedBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
        const pcmChannel0 = decodedBuffer.getChannelData(0);

        // Resample native decoded PCM to 16,000 Hz
        const resampled16k = resampleBuffer(pcmChannel0, decodedBuffer.sampleRate, 16000);

        // Build standard 16 kHz mono WAV
        this.recordedBlob = this.encodeWAV(resampled16k, 16000);
        const audioUrl = URL.createObjectURL(this.recordedBlob);
        this.recordedAudioPlayer.src = audioUrl;
        this.playbackContainer.style.display = 'block';
        this.testRecordBtn.disabled = false;
        await decodeCtx.close();
      } catch (err) {
        console.error('Failed to decode recorded audio:', err);
        alert('Could not decode recorded audio: ' + err.message);
      }
    }
  }

  async testRecordedAudio() {
    if (!this.recordedBlob) return;
    this.testRecordBtn.disabled = true;
    this.testRecordBtn.textContent = 'Processing...';

    try {
      const formData = new FormData();
      formData.append('file', this.recordedBlob, 'user_recording.wav');
      if (this.modelSelect && this.modelSelect.value) {
        formData.append('model_name', this.modelSelect.value);
      }

      const res = await fetch('/api/predict/audio', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `Server error: ${res.status}`);
      }

      this.handlePredictionResult(data, 'User Recording');
    } catch (err) {
      alert('Test failed: ' + err.message);
    } finally {
      this.testRecordBtn.disabled = false;
      this.testRecordBtn.textContent = '⚡ Test Recording';
    }
  }

  /* ========================================================================
     DATASET TEST SAMPLES (INSTANT VERIFICATION)
     ======================================================================== */
  async fetchTestSamples() {
    try {
      const res = await fetch('/api/test_samples');
      const data = await res.json();
      this.sampleSelect.innerHTML = '';
      
      data.samples.forEach((s) => {
        const opt = document.createElement('option');
        opt.value = s.id;
        opt.textContent = `[${s.expected}] ${s.name}`;
        this.sampleSelect.appendChild(opt);
      });
    } catch (err) {
      console.error('Failed to fetch test samples:', err);
    }
  }

  async testDatasetSample() {
    const sampleId = this.sampleSelect.value;
    if (!sampleId) return;

    this.testSampleBtn.disabled = true;
    this.testSampleBtn.textContent = 'Testing...';

    try {
      const formData = new FormData();
      formData.append('sample_id', sampleId);
      if (this.modelSelect && this.modelSelect.value) {
        formData.append('model_name', this.modelSelect.value);
      }

      const res = await fetch('/api/predict/sample', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `Server error: ${res.status}`);
      }

      this.handlePredictionResult(data, `Dataset Sample (${data.filename || sampleId})`);
    } catch (err) {
      alert('Sample test failed: ' + err.message);
    } finally {
      this.testSampleBtn.disabled = false;
      this.testSampleBtn.textContent = 'Run Test';
    }
  }

  /* ========================================================================
     RESULT HANDLING & DETECTION INDICATOR (CORE FEATURE 1 & 2)
     ======================================================================== */
  handlePredictionResult(data, mode) {
    if (!data) return;
    this.totalInferences++;
    
    const confP = typeof data.confidence_proxima === 'number' ? data.confidence_proxima : 0.0;
    const confU = typeof data.confidence_unknown === 'number' ? data.confidence_unknown : 0.0;
    const infTime = typeof data.inference_time_ms === 'number' ? data.inference_time_ms : 0.0;
    const prepTime = typeof data.preprocessing_time_ms === 'number' ? data.preprocessing_time_ms : 0.0;
    
    const now = Date.now();
    const isLive = (mode === 'Live Stream');
    const isGated = Boolean(data.gated);

    // 1. Raw threshold decision (gated silence is never proxima)
    const isProximaRaw = data.is_proxima && (confP >= this.threshold) && !isGated;

    // 2. Temporal Smoothing / Multi-Frame Confirmation:
    // Spoken "PROXIMA" takes ~500ms (2-3 consecutive 250ms evaluation frames).
    // When enabled, requires 2 consecutive frames with confidence >= threshold to trigger.
    // Random noise spikes last only 1 frame and are automatically eliminated!
    let isProximaCandidate = false;
    if (isLive && this.temporalSmoothingEnabled) {
      if (isProximaRaw) {
        this.consecutiveProximaHits++;
        if (this.consecutiveProximaHits >= 2) {
          isProximaCandidate = true;
        }
      } else {
        this.consecutiveProximaHits = 0;
        isProximaCandidate = false;
      }
    } else {
      isProximaCandidate = isProximaRaw;
      if (!isProximaRaw) this.consecutiveProximaHits = 0;
    }

    // 3. Robust Edge-Trigger, Buffer-Flush & Anti-Repeat Debounce Logic:
    let isNewTrigger = false;
    let isProxima = false;

    if (isLive) {
      const elapsed = now - this.lastLiveDetectionTime;
      const inCooldown = (elapsed < this.liveCooldownMs);

      if (inCooldown) {
        // Within refractory lockout window:
        // Strictly prevent count increment or secondary re-triggering on trailing audio
        isNewTrigger = false;
        isProxima = false;
      } else {
        // Cooldown has elapsed. Check for re-arming.
        if (this.liveDetectedState) {
          // Re-arm only after speech drops below threshold (hysteresis guard)
          if (!isProximaRaw || confP < this.threshold) {
            this.liveDetectedState = false;
            this.consecutiveProximaHits = 0;
          }
        }

        // If system is armed and keyword is confirmed:
        if (!this.liveDetectedState && isProximaCandidate) {
          isNewTrigger = true;
          isProxima = true;
          this.liveDetectedState = true;
          this.lastLiveDetectionTime = now;
          this.consecutiveProximaHits = 0;

          // CRITICAL STEP: Flush the circular 2.0s audio buffer with zeroes immediately!
          // This removes the keyword audio from lingering across future 250ms evaluation windows,
          // guaranteeing that 1 spoken utterance generates EXACTLY 1 detection count.
          this.ringBuffer.fill(0);
        }
      }
    } else {
      // Discrete tests (uploaded audio, dataset sample, microphone recording) are always 1-to-1
      isNewTrigger = isProximaCandidate;
      isProxima = isProximaCandidate;
    }

    if (isNewTrigger) {
      this.proximaHits++;
    } else if (!isProxima && !isLive) {
      this.unknownHits++;
    }

    // Update Quick Metrics
    this.metricLatency.textContent = infTime.toFixed(1);
    if (isGated) {
      this.metricPreproc.textContent = `Gated (${data.rms_db || -50} dB)`;
    } else {
      this.metricPreproc.textContent = `Prep: ${prepTime.toFixed(1)} ms`;
    }
    this.metricHits.textContent = this.proximaHits;
    this.metricTotalTests.textContent = `${this.totalInferences} total inferences`;

    // Update CPU Workload & Duty Cycle
    const dutyPct = typeof data.cpu_duty_cycle_pct === 'number' ? data.cpu_duty_cycle_pct : (data.total_latency_ms ? Math.min(100.0, (data.total_latency_ms / 250.0) * 100.0) : 0.0);
    const hostCpu = typeof data.host_cpu_pct === 'number' ? data.host_cpu_pct : 0.0;
    if (this.metricCpuVal) this.metricCpuVal.textContent = dutyPct.toFixed(1);
    if (this.cpuProgressFill) this.cpuProgressFill.style.width = `${Math.min(dutyPct * 10, 100)}%`;
    if (this.metricCpuSub) this.metricCpuSub.textContent = `Duty: ${dutyPct.toFixed(1)}% | Host: ${hostCpu.toFixed(0)}%`;

    // Update Confidence Split Bar
    this.confProximaVal.textContent = `${confP.toFixed(1)}%`;
    this.confUnknownVal.textContent = `${confU.toFixed(1)}%`;
    this.confBarFill.style.width = `${confP.toFixed(1)}%`;

    // GREEN DETECTION INDICATOR ACTIVATION
    if (isNewTrigger) {
      this.activateProximaDetection(confP, infTime);
      this.addHistoryRecord({
        time: new Date().toLocaleTimeString(),
        mode: mode,
        result: 'PROXIMA',
        confProxima: confP,
        confUnknown: confU,
        latency: infTime,
        model: data.model_name || this.currentModelDisplay.textContent
      });
    } else if (!isProxima) {
      if (mode !== 'Live Stream') {
        // Only log discrete tests in history, don't spam table with continuous negative stream
        this.addHistoryRecord({
          time: new Date().toLocaleTimeString(),
          mode: mode,
          result: 'UNKNOWN',
          confProxima: confP,
          confUnknown: confU,
          latency: infTime,
          model: data.model_name || this.currentModelDisplay.textContent
        });
      }
      
      // If we are currently holding a positive detection glow, let it finish its timer
      if (!this.detectionCard.classList.contains('is-detected')) {
        if (this.isLiveListening) {
          const inCooldown = (now - this.lastLiveDetectionTime < this.liveCooldownMs) && this.liveDetectedState;
          if (inCooldown) {
            const remainSec = Math.ceil((this.liveCooldownMs - (now - this.lastLiveDetectionTime)) / 1000);
            this.setListeningState('COOLDOWN', remainSec);
          } else if (isGated) {
            this.setListeningState('GATED');
          } else {
            this.setListeningState('LISTENING');
          }
        } else {
          this.setListeningState('STANDBY');
        }
      }
    }
  }

  activateProximaDetection(confidence, latency) {
    // 1. Activate glowing green hero state
    this.detectionCard.classList.add('is-detected');
    this.listeningStatusPill.className = 'status-indicator-pill state-detected';
    this.listeningStatusText.textContent = '● PROXIMA DETECTED';
    
    this.detectionTitle.textContent = 'PROXIMA DETECTED';
    this.detectionSubtitle.textContent = `Confidence: ${confidence.toFixed(1)}% | Latency: ${latency.toFixed(1)} ms`;

    // 2. Reset hold timer
    if (this.detectionTimer) clearTimeout(this.detectionTimer);
    
    this.detectionTimer = setTimeout(() => {
      this.detectionCard.classList.remove('is-detected');
      if (this.isLiveListening) {
        const now = Date.now();
        const inCooldown = (now - this.lastLiveDetectionTime < this.liveCooldownMs) && this.liveDetectedState;
        if (inCooldown) {
          const remainSec = Math.ceil((this.liveCooldownMs - (now - this.lastLiveDetectionTime)) / 1000);
          this.setListeningState('COOLDOWN', remainSec);
        } else {
          this.setListeningState('LISTENING');
        }
      } else {
        this.setListeningState('STANDBY');
      }
    }, this.detectionHoldMs);
  }

  setListeningState(state, arg) {
    if (state === 'COOLDOWN') {
      this.listeningStatusPill.className = 'status-indicator-pill state-listening';
      this.listeningStatusText.textContent = `● COOLDOWN (${arg || 1}s)`;
      this.detectionTitle.textContent = 'COOLDOWN LOCKOUT ACTIVE';
      this.detectionSubtitle.textContent = 'Wake-word registered. Preventing re-trigger on trailing audio...';
    } else if (state === 'GATED') {
      this.listeningStatusPill.className = 'status-indicator-pill state-listening';
      this.listeningStatusText.textContent = '● SILENCE / GATED';
      this.detectionTitle.textContent = 'LISTENING... (NOISE GATED)';
      this.detectionSubtitle.textContent = 'Ambient room noise filtered out. Speak clearly to trigger.';
    } else if (state === 'LISTENING') {
      this.listeningStatusPill.className = 'status-indicator-pill state-listening';
      this.listeningStatusText.textContent = '● LISTENING';
      this.detectionTitle.textContent = 'LISTENING...';
      this.detectionSubtitle.textContent = 'Say "PROXIMA" clearly into your microphone';
    } else if (state === 'RECORDING') {
      this.listeningStatusPill.className = 'status-indicator-pill state-recording';
      this.listeningStatusText.textContent = '● RECORDING';
      this.detectionTitle.textContent = 'RECORDING AUDIO';
      this.detectionSubtitle.textContent = 'Speak now (saying "PROXIMA" or any speech)';
    } else {
      this.listeningStatusPill.className = 'status-indicator-pill';
      this.listeningStatusText.textContent = '● STANDBY';
      this.detectionTitle.textContent = 'STANDBY';
      this.detectionSubtitle.textContent = 'Start Live Listening, Upload an audio file, or Record audio to test detection';
    }
  }

  /* ========================================================================
     DETECTION HISTORY (CORE FEATURE 6)
     ======================================================================== */
  addHistoryRecord(record) {
    this.historyRecords.unshift(record);
    if (this.historyRecords.length > 50) this.historyRecords.pop();
    this.renderHistoryTable();
  }

  renderHistoryTable() {
    if (!this.historyRecords.length) {
      this.historyTableBody.innerHTML = `
        <tr class="empty-row">
          <td colspan="7">No detections recorded yet. Start live listening or test an audio sample.</td>
        </tr>`;
      this.historyCountBadge.textContent = '0 records';
      return;
    }

    this.historyTableBody.innerHTML = '';
    this.historyRecords.forEach((r) => {
      const tr = document.createElement('tr');
      const isP = (r.result === 'PROXIMA');
      tr.innerHTML = `
        <td>${r.time}</td>
        <td>${r.mode}</td>
        <td><span class="${isP ? 'tag-proxima' : 'tag-unknown'}">${r.result}</span></td>
        <td class="${isP ? 'highlight-green' : ''}">${r.confProxima.toFixed(1)}%</td>
        <td>${r.confUnknown.toFixed(1)}%</td>
        <td>${r.latency.toFixed(1)} ms</td>
        <td style="color: var(--cyan-accent);">${r.model}</td>
      `;
      this.historyTableBody.appendChild(tr);
    });

    this.historyCountBadge.textContent = `${this.historyRecords.length} records`;
  }

  clearHistory() {
    this.historyRecords = [];
    this.renderHistoryTable();
  }

  /* ========================================================================
     AUDIO WAVEFORM DRAWING & ENCODING HELPERS
     ======================================================================== */
  initWaveformCanvas() {
    this.canvasCtx = this.waveformCanvas.getContext('2d');
    this.clearWaveform();
  }

  clearWaveform() {
    const w = this.waveformCanvas.width;
    const h = this.waveformCanvas.height;
    this.canvasCtx.fillStyle = 'rgba(6, 9, 16, 0.8)';
    this.canvasCtx.fillRect(0, 0, w, h);
    
    this.canvasCtx.strokeStyle = 'rgba(0, 210, 255, 0.2)';
    this.canvasCtx.lineWidth = 1;
    this.canvasCtx.beginPath();
    this.canvasCtx.moveTo(0, h / 2);
    this.canvasCtx.lineTo(w, h / 2);
    this.canvasCtx.stroke();
  }

  drawWaveform(samples) {
    const w = this.waveformCanvas.width;
    const h = this.waveformCanvas.height;
    this.canvasCtx.fillStyle = 'rgba(6, 9, 16, 0.4)';
    this.canvasCtx.fillRect(0, 0, w, h);

    const isDetected = this.detectionCard.classList.contains('is-detected');
    this.canvasCtx.strokeStyle = isDetected ? '#00ff88' : '#00d2ff';
    this.canvasCtx.lineWidth = 2;
    this.canvasCtx.beginPath();

    const sliceWidth = w / samples.length;
    let x = 0;

    for (let i = 0; i < samples.length; i++) {
      const v = samples[i] * 1.5;
      const y = (v * (h / 2)) + (h / 2);

      if (i === 0) {
        this.canvasCtx.moveTo(x, y);
      } else {
        this.canvasCtx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    this.canvasCtx.lineTo(w, h / 2);
    this.canvasCtx.stroke();
  }

  encodeWAV(samples, sampleRate = 16000) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
  }
}

// Instantiate client when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.proximaKWS = new ProximaKWSClient();
});
