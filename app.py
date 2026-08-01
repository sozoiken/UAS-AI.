import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

st.set_page_config(page_title="Realtime AI Traffic Dashboard", page_icon="🚦", layout="wide")

# --- CSS Styles ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 20px; border-radius: 12px; color: white; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-container { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 15px; }
    .metric-box { 
        background: #1e293b; padding: 15px; border-radius: 8px; text-align: center; color: white; 
        border-left: 5px solid #00d7ff; flex: 1; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .metric-val { font-size: 28px; font-weight: bold; color: #00d7ff; }
    .metric-lbl { font-size: 14px; color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 2rem;">🚦 ITS - Realtime Traffic Detection Dashboard</h1>
    <p style="margin:0; color:#00d7ff;">Terintegrasi Dalam Satu Layar (Live Detection & Analytics)</p>
</div>
""", unsafe_allow_html=True)

# --- CORE LOGIC CLASSES ---
VEHICLE_CLASS_MAP = {2: 'Mobil', 3: 'Sepeda Motor', 5: 'Bus', 7: 'Truk'}
TARGET_CLASSES = [2, 3, 5, 7]
CLASS_COLORS = {'Mobil': (0, 215, 255), 'Sepeda Motor': (255, 144, 30), 'Bus': (50, 205, 50), 'Truk': (50, 50, 240)}
HEX_COLORS = {'Mobil': '#00d7ff', 'Sepeda Motor': '#ff901e', 'Bus': '#32cd32', 'Truk': '#f03232', 'Total Volume': '#8b5cf6'}

def _refine_class(cls_id, cls_name, bbox, conf):
    x1, y1, x2, y2 = bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    area, ar = w * h, w / h
    
    if cls_name == 'Bus':
        if area < 5000: return 'Mobil', 2, conf
        elif ar < 1.2 and w < 120: return 'Truk', 7, conf
    if cls_name == 'Truk':
        if area < 3000: return 'Mobil', 2, conf
    if cls_name == 'Mobil':
        if area < 2500 and w < 70: return 'Sepeda Motor', 3, conf
        if area < 4000 and w < 80 and h > w * 0.85: return 'Sepeda Motor', 3, conf
    if cls_name == 'Sepeda Motor':
        if area > 16000 or (w > 150 and h > 75): return 'Mobil', 2, conf
        
    return cls_name, cls_id, conf

def _nms(detections, iou_thr=0.45):
    if len(detections) <= 1: return detections
    boxes  = np.array([d['bbox'] for d in detections], dtype=float)
    scores = np.array([d['confidence'] for d in detections])
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas  = (x2-x1+1)*(y2-y1+1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0,xx2-xx1+1)*np.maximum(0,yy2-yy1+1)
        iou   = inter/(areas[i]+areas[order[1:]]-inter)
        order = order[np.where(iou<=iou_thr)[0]+1]
    return [detections[i] for i in keep]

class VehicleDetector:
    def __init__(self, conf=0.25, iou=0.45):
        self.conf = conf
        self.iou = iou
        self.model = YOLO('yolov8n.pt') if YOLO_AVAILABLE else None
    def detect(self, frame):
        if frame is None or frame.size == 0: return []
        dets = []
        if self.model:
            res = self.model.predict(source=frame, conf=self.conf, iou=self.iou, classes=TARGET_CLASSES, verbose=False, agnostic_nms=True)
            if res and len(res[0].boxes) > 0:
                for b in res[0].boxes:
                    x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
                    if (x2-x1) < 18 or (y2-y1) < 14: continue
                    cid = int(b.cls[0])
                    name = VEHICLE_CLASS_MAP.get(cid, 'Kendaraan')
                    conf_val = float(b.conf[0])
                    name, cid, conf_val = _refine_class(cid, name, (x1,y1,x2,y2), conf_val)
                    dets.append({'bbox':(x1,y1,x2,y2), 'class_name':name, 'confidence':conf_val, 'class_id':cid})
        return _nms(dets, self.iou)

class VehicleTracker:
    def __init__(self, max_dist=120, max_gone=25):
        self.next_id = 1
        self.objects = {}; self.vels = {}; self.data = {}; self.gone = {}
        self.max_dist = max_dist; self.max_gone = max_gone
        self.counts = {'Mobil':0, 'Sepeda Motor':0, 'Bus':0, 'Truk':0}

    def _reg(self, cen, det):
        oid = self.next_id
        self.objects[oid] = cen; self.vels[oid] = (0.0, 0.0)
        self.data[oid] = {'class_name': det['class_name'], 'counted': False, 'cooldown': 0, 'hist': [cen]}
        self.gone[oid] = 0
        self.next_id += 1

    def _del(self, oid):
        for d in [self.objects, self.vels, self.data, self.gone]: d.pop(oid, None)

    def update(self, dets, line=None):
        tracked = []
        for oid in self.data:
            if self.data[oid]['cooldown'] > 0: self.data[oid]['cooldown'] -= 1
        if not dets:
            for oid in list(self.gone):
                self.gone[oid] += 1
                if self.gone[oid] > self.max_gone: self._del(oid)
            return tracked

        inp = [((d['bbox'][0]+d['bbox'][2])//2, (d['bbox'][1]+d['bbox'][3])//2) for d in dets]
        if not self.objects:
            for i, d in enumerate(dets):
                self._reg(inp[i], d)
                dc = d.copy(); dc['object_id'] = self.next_id-1
                tracked.append(dc)
            return tracked

        oids = list(self.objects)
        pred = [(int(self.objects[oid][0] + self.vels[oid][0]), int(self.objects[oid][1] + self.vels[oid][1])) for oid in oids]
        D = np.linalg.norm(np.array(pred)[:,None] - np.array(inp)[None,:], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        ur, uc = set(), set()

        for r, c in zip(rows, cols):
            if r in ur or c in uc or D[r,c] > self.max_dist: continue
            oid = oids[r]
            prev = self.objects[oid]
            curr = inp[c]

            vx = 0.6*(curr[0]-prev[0]) + 0.4*self.vels[oid][0]
            vy = 0.6*(curr[1]-prev[1]) + 0.4*self.vels[oid][1]
            self.vels[oid] = (vx, vy)
            self.objects[oid] = curr
            self.gone[oid] = 0
            self.data[oid]['class_name'] = dets[c]['class_name']
            
            if line: self._check(oid, prev, curr, line)

            dc = dets[c].copy(); dc['object_id'] = oid
            tracked.append(dc)
            ur.add(r); uc.add(c)

        for r, oid in enumerate(oids):
            if r not in ur:
                self.gone[oid] += 1
                if self.gone[oid] > self.max_gone: self._del(oid)

        for c in range(len(inp)):
            if c not in uc:
                self._reg(inp[c], dets[c])
                dc = dets[c].copy(); dc['object_id'] = self.next_id-1
                tracked.append(dc)
        return tracked

    def _check(self, oid, p1, p2, line):
        if self.data[oid]['counted'] or self.data[oid]['cooldown'] > 0: return
        l1, l2 = line
        crossed = False
        if l1[1] == l2[1]: # horizontal line
            ly = l1[1]
            minx, maxx = min(l1[0],l2[0]), max(l1[0],l2[0])
            in_x = (minx <= p2[0] <= maxx) or (minx <= p1[0] <= maxx)
            crossed = in_x and ((p1[1]<ly and p2[1]>=ly) or (p1[1]>ly and p2[1]<=ly))

        if crossed:
            cls = self.data[oid]['class_name']
            if cls in self.counts: self.counts[cls] += 1
            self.data[oid]['counted'] = True
            self.data[oid]['cooldown'] = 15

def draw_boxes(frame, dets, line=None, time_str="00:00"):
    out = frame.copy()
    if line:
        cv2.line(out, line[0], line[1], (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(out, 'ROI COUNTER', (line[0][0]+8, line[0][1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    
    # Overlay Waktu Video
    cv2.putText(out, f"Waktu: {time_str}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    for d in dets:
        x1,y1,x2,y2 = d['bbox']
        cls = d['class_name']
        conf = d.get('confidence', 0.0)
        oid = d.get('object_id', -1)
        col = CLASS_COLORS.get(cls, (0,255,0))
        
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
        lbl = f"ID:{oid} {cls} {int(conf*100)}%"
        (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0,y1-th-7)), (x1+tw+8, y1), col, -1)
        cv2.putText(out, lbl, (x1+4, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,10), 1, cv2.LINE_AA)
    return out

# --- SESSION STATE INITIALIZATION ---
if 'status' not in st.session_state:
    st.session_state.status = 'STOPPED'  # States: STOPPED, RUNNING, PAUSED
if 'current_frame_idx' not in st.session_state:
    st.session_state.current_frame_idx = 0
if 'time_series' not in st.session_state:
    st.session_state.time_series = []
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None
if 'tracker' not in st.session_state:
    st.session_state.tracker = VehicleTracker(max_dist=120)

# --- CALLBACK FUNCTIONS ---
def btn_start():
    st.session_state.status = 'RUNNING'
    st.session_state.current_frame_idx = 0
    st.session_state.time_series = []
    st.session_state.tracker = VehicleTracker(max_dist=120) # Reset AI tracker
    st.session_state.last_frame = None

def btn_pause():
    st.session_state.status = 'PAUSED'

def btn_resume():
    st.session_state.status = 'RUNNING'

def btn_stop():
    st.session_state.status = 'STOPPED'

# ============================
# Sidebar Configuration
# ============================
st.sidebar.header("⚙️ Konfigurasi")

conf_thresh = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05
)

# ============================
# Video Path (Portable)
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(
    BASE_DIR,
    "data",
    "cctv_youtube_indonesia.mp4"
)

# Menampilkan lokasi video (read only)
st.sidebar.text_input(
    "Video Input",
    value=video_path,
    disabled=True
)

# Cek apakah video tersedia
if not os.path.exists(video_path):
    st.sidebar.error("❌ File video tidak ditemukan!")

st.sidebar.markdown("---")
st.sidebar.subheader("🕹️ Kontrol Pemutaran")

# Dynamic Buttons based on status
if st.session_state.status == 'STOPPED':
    st.sidebar.button("🚀 Mulai Deteksi", on_click=btn_start, use_container_width=True)
elif st.session_state.status == 'RUNNING':
    st.sidebar.button("⏸️ Jeda (Pause)", on_click=btn_pause, use_container_width=True)
    st.sidebar.button("🛑 Berhenti Total", on_click=btn_stop, use_container_width=True)
elif st.session_state.status == 'PAUSED':
    st.sidebar.button("▶️ Lanjut (Resume)", on_click=btn_resume, use_container_width=True)
    st.sidebar.button("🛑 Berhenti Total", on_click=btn_stop, use_container_width=True)
    st.sidebar.button("🔄 Ulang dari Awal", on_click=btn_start, use_container_width=True)

# --- UI Main Layout Placeholders ---
col_video, col_stats = st.columns([2.5, 1.5])
with col_video:
    st.subheader("📹 Live Video Stream & Detection")
    video_placeholder = st.empty()
with col_stats:
    st.subheader("📊 Live Traffic Counter")
    metrics_placeholder = st.empty()

st.markdown("---")
st.markdown("## Analisis Data & Visualisasi Pola Lalu Lintas")

col_bar_donut, col_line = st.columns([1, 1.2])
with col_bar_donut:
    col_bar, col_donut = st.columns(2)
    bar_placeholder = col_bar.empty()
    donut_placeholder = col_donut.empty()

line_placeholder = col_line.empty()

st.markdown("### 📋 Ringkasan Statistik Volume Lalu Lintas")
table_placeholder = st.empty()

st.markdown("### 💡 Interpretasi Pola Lalu Lintas & Rekomendasi Kebijakan:")
st.markdown("""
1. **Dominasi Kendaraan**: Sepeda motor dan mobil penumpang merupakan moda transportasi dominan di ruas jalan perkotaan Indonesia (>80% dari total volume).
2. **Perilaku Arus Lalu Lintas**: Arus kendaraan mengalami lonjakan (*peak*) pada interval pertengahan pengamatan, yang mencerminkan karakteristik puncak jam sibuk (*rush hour*).
3. **Rekomendasi Manajerial (ITS)**:
   - Optimalisasi durasi sinyal lampu lalu lintas adaptif berdasarkan *real-time count*.
   - Pemberlakuan jalur khusus untuk kendaraan berat (Bus & Truk) guna mencegah kemacetan dan kecelakaan.
""")

# --- Render Function ---
def render_dashboard(frame_rgb, counts, time_series_data, unique_key_suffix):
    if frame_rgb is not None:
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
    c_mobil = counts.get('Mobil', 0)
    c_motor = counts.get('Sepeda Motor', 0)
    c_bus = counts.get('Bus', 0)
    c_truk = counts.get('Truk', 0)
    c_tot = sum(counts.values())
    
    html_metrics = f"""
    <div class="metric-container">
        <div class="metric-box" style="border-left-color: #00d7ff;">
            <div class="metric-val">{c_mobil}</div><div class="metric-lbl">🚗 Mobil</div>
        </div>
        <div class="metric-box" style="border-left-color: #ff901e;">
            <div class="metric-val">{c_motor}</div><div class="metric-lbl">🛵 Motor</div>
        </div>
    </div>
    <div class="metric-container">
        <div class="metric-box" style="border-left-color: #32cd32;">
            <div class="metric-val">{c_bus}</div><div class="metric-lbl">🚌 Bus</div>
        </div>
        <div class="metric-box" style="border-left-color: #f03232;">
            <div class="metric-val">{c_truk}</div><div class="metric-lbl">🚚 Truk</div>
        </div>
    </div>
    <div class="metric-container">
        <div class="metric-box" style="border-left-color: #a855f7;">
            <div class="metric-val" style="color: #a855f7;">{c_tot}</div><div class="metric-lbl">TOTAL KENDARAAN</div>
        </div>
    </div>
    """
    metrics_placeholder.markdown(html_metrics, unsafe_allow_html=True)
    
    if len(time_series_data) > 0:
        df = pd.DataFrame(time_series_data)
        
        # 1. Bar Chart
        df_bar = pd.DataFrame({
            'Kategori': ['Mobil', 'Sepeda Motor', 'Bus', 'Truk'],
            'Jumlah': [c_mobil, c_motor, c_bus, c_truk]
        })
        fig_bar = px.bar(df_bar, x='Kategori', y='Jumlah', text='Jumlah', 
                         title="Distribusi Jumlah Kendaraan",
                         color='Kategori', color_discrete_map=HEX_COLORS)
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300, showlegend=False)
        bar_placeholder.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{unique_key_suffix}")

        # 2. Donut Chart
        fig_donut = px.pie(df_bar, values='Jumlah', names='Kategori', hole=0.5,
                           title="Proporsi (%) Kategori",
                           color='Kategori', color_discrete_map=HEX_COLORS)
        fig_donut.update_traces(textinfo='percent+label')
        fig_donut.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300, showlegend=False)
        donut_placeholder.plotly_chart(fig_donut, use_container_width=True, key=f"donut_{unique_key_suffix}")

        # 3. Line Chart
        fig_line = go.Figure()
        for col in ['Mobil', 'Sepeda Motor', 'Bus', 'Truk', 'Total Volume']:
            fig_line.add_trace(go.Scatter(x=df['Timestamp'], y=df[col], mode='lines+markers', name=col,
                                          line=dict(color=HEX_COLORS.get(col, '#ffffff'), width=2)))
        fig_line.update_layout(title="Dinamika Perubahan Volume Kendaraan",
                               xaxis_title="Waktu Pengamatan (Interval)",
                               yaxis_title="Volume Kendaraan",
                               margin=dict(l=20, r=20, t=40, b=20), height=300,
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        line_placeholder.plotly_chart(fig_line, use_container_width=True, key=f"line_{unique_key_suffix}")
        
        # 4. Table Statistics
        stats_data = []
        for cat in ['Mobil', 'Sepeda Motor', 'Bus', 'Truk', 'Total Volume']:
            mean_val = df[cat].mean()
            median_val = df[cat].median()
            std_val = df[cat].std() if len(df) > 1 else 0
            min_val = df[cat].min()
            max_val = df[cat].max()
            
            if max_val > 0:
                peak_idx = df[cat].idxmax()
                peak_time = df.loc[peak_idx, 'Timestamp']
            else:
                peak_time = "00:00"

            stats_data.append({
                "Kategori": cat,
                "Mean": round(mean_val, 2),
                "Median": round(median_val, 1),
                "Std": round(std_val, 2),
                "Min": int(min_val),
                "Peak": int(max_val),
                "Peak Time": peak_time
            })
        df_stats = pd.DataFrame(stats_data)
        table_placeholder.dataframe(df_stats, use_container_width=True)

# --- VIDEO PROCESSING LOOP ---
if st.session_state.status == 'RUNNING':
    if not YOLO_AVAILABLE:
        st.error("Ultralytics YOLO tidak terinstall. Jalankan `pip install ultralytics`")
        st.session_state.status = 'STOPPED'
    elif not os.path.exists(video_path):
        st.error(f"File video tidak ditemukan di path: {video_path}")
        st.session_state.status = 'STOPPED'
    else:
        cap = cv2.VideoCapture(video_path)
        # Seek ke frame terakhir yang disimpan
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_idx)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        line_y = int(height * 0.55)
        counting_line = ((int(width * 0.05), line_y), (int(width * 0.95), line_y))
        
        detector = VehicleDetector(conf=conf_thresh)
        tracker = st.session_state.tracker # Gunakan tracker yang tersimpan
        
        while cap.isOpened() and st.session_state.status == 'RUNNING':
            ret, frame = cap.read()
            if not ret:
                st.info("Pemrosesan Video Selesai.")
                st.session_state.status = 'STOPPED'
                st.rerun()
            
            st.session_state.current_frame_idx += 1
            
            # Konversi Frame ke Menit:Detik
            total_sec = st.session_state.current_frame_idx // int(fps_video)
            mins = int(total_sec // 60)
            secs = int(total_sec % 60)
            time_str = f"{mins:02d}:{secs:02d}"
            
            # Deteksi & Tracking
            dets = detector.detect(frame)
            tracked_objs = tracker.update(dets, line=counting_line)
            annotated_frame = draw_boxes(frame, tracked_objs, line=counting_line, time_str=time_str)
            
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.session_state.last_frame = rgb_frame
            
            c_tot = sum(tracker.counts.values())
            
            # Update visualisasi setiap 10 Frame
            if st.session_state.current_frame_idx % 10 == 0:
                st.session_state.time_series.append({
                    "Timestamp": time_str,
                    "Mobil": tracker.counts['Mobil'],
                    "Sepeda Motor": tracker.counts['Sepeda Motor'],
                    "Bus": tracker.counts['Bus'],
                    "Truk": tracker.counts['Truk'],
                    "Total Volume": c_tot
                })
                render_dashboard(rgb_frame, tracker.counts, st.session_state.time_series, unique_key_suffix=st.session_state.current_frame_idx)
            else:
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        cap.release()

# --- RENDER STATIC STATE (PAUSED OR STOPPED) ---
if st.session_state.status in ['PAUSED', 'STOPPED']:
    if st.session_state.status == 'PAUSED':
        st.warning(f"⏸️ Video dijeda pada frame ke-{st.session_state.current_frame_idx}. Klik 'Lanjut' di sidebar untuk meneruskan.")
    elif st.session_state.status == 'STOPPED' and st.session_state.last_frame is not None:
        st.success("✅ Pemrosesan video telah dihentikan atau selesai.")

    if st.session_state.last_frame is not None:
        render_dashboard(st.session_state.last_frame, st.session_state.tracker.counts, st.session_state.time_series, unique_key_suffix="static")
