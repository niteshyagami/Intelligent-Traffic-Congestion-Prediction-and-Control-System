
    const API = window.location.origin;
    let predChart = null, signalChart = null, trainingChart = null;
    let webcamStream = null;
    let videoFilePlayer = null;
    let videoLoopId = null;

    // ── Tab Switching ──
    function switchTab(tab) {
        ['prediction', 'signal', 'analytics'].forEach(t => {
            document.getElementById(`panel-${t}`).style.display = t === tab ? '' : 'none';
            const btn = document.getElementById(`tab-${t}`);
            btn.className = t === tab 
                ? 'tab-active px-5 py-2.5 rounded-xl text-sm font-semibold transition-all'
                : 'tab-inactive px-5 py-2.5 rounded-xl text-sm font-semibold transition-all';
        });
        if (tab === 'analytics') loadModelInfo();
    }

    // ── Health Check ──
    async function checkHealth() {
        try {
            const r = await fetch(`${API}/health`);
            const d = await r.json();
            const el = document.getElementById('connStatus');
            el.innerHTML = `<span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span> Connected`;
        } catch {
            document.getElementById('connStatus').innerHTML = '<span class="w-2 h-2 rounded-full bg-red-500"></span> Disconnected';
        }
    }

    // ── Prediction ──
    async function runPrediction() {
        const date = document.getElementById('predDate').value;
        const hour = document.getElementById('predHour').value;
        const minute = document.getElementById('predMinute').value;
        if (!date) { alert('Please select a date'); return; }

        const btn = document.getElementById('btnPredict');
        btn.textContent = 'Predicting...';
        btn.disabled = true;

        const hh = String(hour).padStart(2, '0');
        const mm = String(minute).padStart(2, '0');
        const datetime = `${date} ${hh}:${mm}:00`;
        const intId = document.getElementById('predIntersection').value;

        try {
            const r = await fetch(`${API}/api/v1/predict`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ target_datetime: datetime, intersection_id: intId || null })
            });
            const data = await r.json();
            renderPredictions(data.predictions);
        } catch(e) {
            alert('Prediction failed: ' + e.message);
        } finally {
            btn.textContent = 'Predict Congestion';
            btn.disabled = false;
        }
    }

    function renderPredictions(predictions) {
        const container = document.getElementById('predResults');
        container.innerHTML = '';

        const chartLabels = [];
        const chartDatasets = {};
        const laneColors = { Lane_N: '#facc15', Lane_E: '#22c55e', Lane_S: '#3b82f6', Lane_W: '#ef4444' };

        predictions.forEach(p => {
            const badgeClass = p.overall_congestion === 'High' ? 'badge-high' : p.overall_congestion === 'Medium' ? 'badge-medium' : 'badge-low';

            let lanesHTML = '';
            p.lanes.forEach(l => {
                const lBadge = l.predicted_congestion === 'High' ? 'badge-high' : l.predicted_congestion === 'Medium' ? 'badge-medium' : 'badge-low';
                const maxGreen = 45;
                const barWidth = Math.min(100, (l.predicted_vehicle_count / 60) * 100);
                lanesHTML += `
                    <div class="flex items-center justify-between py-1 border-b border-white/5 last:border-0">
                        <span class="text-xs text-white/50 w-16">${l.lane_id.replace('Lane_', '')}</span>
                        <div class="flex-1 mx-2 h-2 bg-white/10 rounded-full overflow-hidden">
                            <div class="h-full progress-bar rounded-full" style="width:${barWidth}%; background: ${laneColors[l.lane_id] || '#888'}"></div>
                        </div>
                        <span class="text-xs text-white/60 w-10 text-right">${l.predicted_vehicle_count}v</span>
                        <span class="text-xs ${lBadge} px-1.5 py-0.5 rounded ml-2">${l.predicted_congestion}</span>
                    </div>`;

                if (!chartDatasets[l.lane_id]) chartDatasets[l.lane_id] = [];
                chartDatasets[l.lane_id].push(l.predicted_vehicle_count);
            });

            chartLabels.push(p.intersection_name);

            container.innerHTML += `
                <div class="glass rounded-2xl p-4 animate-in lane-card">
                    <div class="flex items-center justify-between mb-2">
                        <div>
                            <h3 class="text-sm font-semibold text-white/90">${p.intersection_name}</h3>
                            <p class="text-xs text-white/40">${p.city} | ${p.timestamp}</p>
                        </div>
                        <span class="${badgeClass} px-2.5 py-1 rounded-full text-xs font-bold">${p.overall_congestion}</span>
                    </div>
                    <div class="mt-2">${lanesHTML}</div>
                </div>`;
        });

        // Update chart
        document.getElementById('predChartContainer').style.display = '';
        const ctx = document.getElementById('predChart').getContext('2d');
        if (predChart) predChart.destroy();

        const datasets = Object.entries(chartDatasets).map(([lane, data]) => ({
            label: lane.replace('Lane_', 'Lane '),
            data: data,
            backgroundColor: laneColors[lane] + '80',
            borderColor: laneColors[lane],
            borderWidth: 1,
        }));

        predChart = new Chart(ctx, {
            type: 'bar',
            data: { labels: chartLabels, datasets },
            options: {
                responsive: true,
                plugins: { legend: { labels: { color: 'rgba(255,255,255,0.6)', font: { size: 11 } } } },
                scales: {
                    x: { ticks: { color: 'rgba(255,255,255,0.4)', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: 'rgba(255,255,255,0.4)' }, grid: { color: 'rgba(255,255,255,0.05)' }, title: { display: true, text: 'Vehicle Count', color: 'rgba(255,255,255,0.4)' } }
                }
            }
        });
    }

    // ── Signal Control ──
    function renderSignalCards(data) {
        const container = document.getElementById('signalCards');
        const lanes = data.lanes || {};
        const laneColors = { Lane_N: 'yellow', Lane_E: 'green', Lane_S: 'blue', Lane_W: 'red' };
        const laneLabels = { Lane_N: 'North', Lane_E: 'East', Lane_S: 'South', Lane_W: 'West' };

        container.innerHTML = `
            <div class="text-center mb-3">
                <span class="text-2xl font-bold text-white/90">${data.total_vehicles}</span>
                <span class="text-xs text-white/40 ml-1">total vehicles</span>
                <span class="ml-3 ${data.congestion_level === 'High' ? 'badge-high' : data.congestion_level === 'Medium' ? 'badge-medium' : 'badge-low'} px-2.5 py-1 rounded-full text-xs font-bold">${data.congestion_level}</span>
            </div>`;

        Object.entries(lanes).forEach(([lid, l]) => {
            const isGreen = l.phase === 'GREEN';
            const total = l.green_duration + l.red_duration;
            const greenPct = (l.green_duration / Math.max(total, 1)) * 100;

            container.innerHTML += `
                <div class="glass-hover rounded-xl p-3 transition-all animate-in">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center gap-2">
                            <div class="w-4 h-4 rounded-full ${isGreen ? 'signal-green' : 'signal-red'}"></div>
                            <span class="text-sm font-semibold">${laneLabels[lid] || lid}</span>
                        </div>
                        <span class="text-xs text-white/50">${l.vehicle_count} vehicles</span>
                    </div>
                    <div class="mt-2 flex items-center gap-2">
                        <div class="flex-1 h-3 bg-white/10 rounded-full overflow-hidden">
                            <div class="h-full rounded-full progress-bar" style="width:${greenPct}%; background: linear-gradient(90deg, #00ff88, #00cc66);"></div>
                        </div>
                        <span class="text-xs text-green-400 font-mono">${l.green_duration}s</span>
                        <span class="text-xs text-red-400 font-mono">${l.red_duration}s</span>
                    </div>
                </div>`;
        });

        // Update signal chart
        document.getElementById('signalChartContainer').style.display = '';
        const ctx = document.getElementById('signalChart').getContext('2d');
        if (signalChart) signalChart.destroy();

        const laneIds = Object.keys(lanes);
        signalChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: laneIds.map(l => laneLabels[l] || l),
                datasets: [
                    { label: 'Green (s)', data: laneIds.map(l => lanes[l].green_duration), backgroundColor: 'rgba(0,255,136,0.6)', borderColor: '#00ff88', borderWidth: 1 },
                    { label: 'Red (s)', data: laneIds.map(l => lanes[l].red_duration), backgroundColor: 'rgba(255,68,68,0.6)', borderColor: '#ff4444', borderWidth: 1 },
                ]
            },
            options: {
                responsive: true,
                indexAxis: 'y',
                plugins: { legend: { labels: { color: 'rgba(255,255,255,0.6)' } } },
                scales: {
                    x: { stacked: true, ticks: { color: 'rgba(255,255,255,0.4)' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { stacked: true, ticks: { color: 'rgba(255,255,255,0.4)' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    }

    async function submitManualCounts() {
        const counts = {
            Lane_N: parseInt(document.getElementById('manualN').value) || 0,
            Lane_E: parseInt(document.getElementById('manualE').value) || 0,
            Lane_S: parseInt(document.getElementById('manualS').value) || 0,
            Lane_W: parseInt(document.getElementById('manualW').value) || 0,
        };

        try {
            const r = await fetch(`${API}/api/v1/signal-update`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ intersection_id: 'INT_01', intersection_name: 'Manual Input', lane_counts: counts })
            });
            const data = await r.json();
            renderSignalCards(data);
        } catch(e) {
            alert('Signal update failed: ' + e.message);
        }
    }
    async function handleFileUpload(file) {
        if (!file) return;
        document.getElementById('dropZoneText').textContent = 'Processing: ' + file.name;
        document.getElementById('detectionInfo').textContent = 'Loading file...';

        if (file.type.startsWith('image/')) {
            // Handle image files
            const reader = new FileReader();
            reader.onload = async (e) => {
                const img = new Image();
                img.onload = async () => {
                    const canvas = document.getElementById('videoCanvas');
                    // Scale to max 640px wide for performance
                    const scale = Math.min(1, 640 / img.width);
                    canvas.width = Math.round(img.width * scale);
                    canvas.height = Math.round(img.height * scale);
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    await sendFrameForDetection(canvas);
                    document.getElementById('dropZoneText').textContent = 'Uploaded: ' + file.name + ' (click to change)';
                };
                img.onerror = () => { alert('Could not load image file'); };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);

        } else if (file.type.startsWith('video/')) {
            // Handle video files — play inline and loop frames dynamically
            const video = document.createElement('video');
            video.preload = 'auto';
            video.muted = true;
            video.loop = true;
            const url = URL.createObjectURL(file);
            video.src = url;

            if (videoFilePlayer) videoFilePlayer.pause();
            if (videoLoopId) clearTimeout(videoLoopId);
            videoFilePlayer = video;

            video.onloadeddata = async () => {
                video.play();
                document.getElementById('dropZoneText').textContent = 'Live Analysis: ' + file.name + ' (click to change or stop)';
                
                const canvas = document.getElementById('videoCanvas');
                const ctx = canvas.getContext('2d');
                
                const captureAndDetect = async () => {
                    if (video.paused || video.ended) return;
                    
                    const scale = Math.min(1, 640 / video.videoWidth);
                    canvas.width = Math.round(video.videoWidth * scale);
                    canvas.height = Math.round(video.videoHeight * scale);
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    await sendFrameForDetection(canvas);
                    
                    videoLoopId = setTimeout(captureAndDetect, 1500); // Poll every 1.5s
                };
                captureAndDetect();
            };
            video.onerror = () => {
                alert('Could not load video file. Try a different format (MP4 works best).');
                URL.revokeObjectURL(url);
            };
        } else {
            alert('Unsupported file type: ' + file.type + '. Please upload an image (JPG, PNG) or video (MP4).');
        }
    }

    async function sendFrameForDetection(canvas) {
        const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
        const b64 = dataUrl.split(',')[1];
        const ctx = canvas.getContext('2d');

        try {
            const mode = document.getElementById('detectionMode') ? document.getElementById('detectionMode').value : 'intersection';
            const r = await fetch(`${API}/api/v1/detect`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ image_base64: b64, intersection_id: 'INT_01', mode: mode })
            });
            const data = await r.json();

            if (data.annotated_image_base64) {
                const annImg = new Image();
                annImg.onload = () => { ctx.drawImage(annImg, 0, 0, canvas.width, canvas.height); };
                annImg.src = 'data:image/jpeg;base64,' + data.annotated_image_base64;
            }

            document.getElementById('detectionInfo').textContent = 
                `Detected: ${data.total_vehicles} vehicles | N:${data.lane_counts.Lane_N || 0} E:${data.lane_counts.Lane_E || 0} S:${data.lane_counts.Lane_S || 0} W:${data.lane_counts.Lane_W || 0}`;

            const sr = await fetch(`${API}/api/v1/signals`);
            const sd = await sr.json();
            if (sd.intersections && sd.intersections.length > 0) renderSignalCards(sd.intersections[0]);
        } catch(e) { 
            alert('Detection failed: ' + e.message);
            document.getElementById('detectionInfo').textContent = 'Detection failed — check if server is running';
        }
    }

    async function startWebcam() {
        try {
            // Explicitly request ideal constraints to avoid hardware allocation issues
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } } });
            webcamStream = stream;
            const video = document.createElement('video');
            video.srcObject = stream;
            video.play();

            const canvas = document.getElementById('videoCanvas');
            const ctx = canvas.getContext('2d');

            const captureFrame = async () => {
                if (!webcamStream) return;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
                const b64 = dataUrl.split(',')[1];

                try {
                    const mode = document.getElementById('detectionMode') ? document.getElementById('detectionMode').value : 'intersection';
                    const r = await fetch(`${API}/api/v1/detect`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ image_base64: b64, intersection_id: 'INT_01', mode: mode })
                    });
                    const data = await r.json();
                    if (data.annotated_image_base64) {
                        const img = new Image();
                        img.onload = () => { ctx.drawImage(img, 0, 0, canvas.width, canvas.height); };
                        img.src = 'data:image/jpeg;base64,' + data.annotated_image_base64;
                    }
                    document.getElementById('detectionInfo').textContent = 
                        `Live | Vehicles: ${data.total_vehicles} | N:${data.lane_counts.Lane_N || 0} E:${data.lane_counts.Lane_E || 0} S:${data.lane_counts.Lane_S || 0} W:${data.lane_counts.Lane_W || 0}`;

                    const sr = await fetch(`${API}/api/v1/signals`);
                    const sd = await sr.json();
                    if (sd.intersections && sd.intersections.length > 0) renderSignalCards(sd.intersections[0]);
                } catch {}

                if (webcamStream) setTimeout(captureFrame, 2000);
            };
            setTimeout(captureFrame, 1000);
        } catch(e) {
            console.error('Webcam error:', e);
            if (e.name === 'NotReadableError' || e.message.includes('busy')) {
                 alert('Error: Your webcam is currently being used by another app (like Zoom, Teams, or another browser tab).\n\nPlease close the other app and click "Use Web Camera" again!');
            } else if (e.name === 'NotAllowedError' || e.message.includes('Permission denied')) {
                 alert('Error: Your browser is blocking camera access! \n\nClick the 🔒 icon next to "localhost:8000" at the top of Chrome, allow the Camera, and refresh the page. We cannot open it unless you click Allow.');
            } else if (e.name === 'NotFoundError') {
                 alert('Error: No webcam was detected on this computer! Please plug in a webcam.');
            } else {
                 alert('Webcam access failed: ' + e.message);
            }
        }
    }

    // ── Analytics ──
    async function loadModelInfo() {
        try {
            const r = await fetch(`${API}/api/v1/model-info`);
            const data = await r.json();
            const el = document.getElementById('modelInfo');

            if (data.status === 'not_trained') {
                el.innerHTML = `<p class="text-amber-400">Model not trained. Run: <code class="bg-white/10 px-2 py-0.5 rounded text-xs">python scripts/train_model.py</code></p>`;
                return;
            }

            el.innerHTML = `
                <div class="grid grid-cols-2 gap-2">
                    <div class="p-2 bg-white/5 rounded-lg"><span class="text-xs text-white/40 block">Model</span><span class="text-sm font-semibold">${data.model_type}</span></div>
                    <div class="p-2 bg-white/5 rounded-lg"><span class="text-xs text-white/40 block">Accuracy</span><span class="text-sm font-semibold text-green-400">${(data.best_val_acc * 100).toFixed(2)}%</span></div>
                    <div class="p-2 bg-white/5 rounded-lg"><span class="text-xs text-white/40 block">Features</span><span class="text-sm font-semibold">${data.input_dim}</span></div>
                    <div class="p-2 bg-white/5 rounded-lg"><span class="text-xs text-white/40 block">Classes</span><span class="text-sm font-semibold">${data.class_names.join(', ')}</span></div>
                    <div class="p-2 bg-white/5 rounded-lg"><span class="text-xs text-white/40 block">Seq Length</span><span class="text-sm font-semibold">${data.seq_len}</span></div>
                    <div class="p-2 bg-white/5 rounded-lg"><span class="text-xs text-white/40 block">Epochs</span><span class="text-sm font-semibold">${data.epochs_trained}</span></div>
                </div>
                <div class="mt-2 text-xs text-white/40">
                    <strong>Features:</strong> ${data.feature_names.join(', ')}
                </div>`;

            // Training chart placeholder
            document.getElementById('trainingChartStatus').textContent = 'Training history from model metadata';
        } catch(e) {
            document.getElementById('modelInfo').innerHTML = `<p class="text-red-400">Error loading model info</p>`;
        }
    }

    // ── Init ──
    // Set date and time to current local values (fully dynamic, no static defaults)
    const now = new Date();
    const yyyy = now.getFullYear();
    const mm = String(now.getMonth() + 1).padStart(2, '0');
    const dd = String(now.getDate()).padStart(2, '0');
    document.getElementById('predDate').value = `${yyyy}-${mm}-${dd}`;
    document.getElementById('predHour').value = String(now.getHours());
    // Set minute to nearest 5
    const nearestMin = Math.round(now.getMinutes() / 5) * 5;
    document.getElementById('predMinute').value = String(nearestMin >= 60 ? 55 : nearestMin);

    // File upload — click handler
    const dropZone = document.getElementById('dropZone');
    const videoInput = document.getElementById('videoInput');

    dropZone.addEventListener('click', () => videoInput.click());
    videoInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) handleFileUpload(e.target.files[0]);
    });

    // Drag-drop handler
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault(); e.stopPropagation(); dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleFileUpload(file);
    });

    checkHealth();
    setInterval(checkHealth, 10000);
    
