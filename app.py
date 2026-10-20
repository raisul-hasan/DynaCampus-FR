"""
DynaCampus-FR — Research Dashboard
===================================
Streamlit application for face enrollment, recognition, and live video inference.
Demonstrates training-free dynamic enrollment with the full pipeline:
  SCRFD detection → alignment → embedding → gallery cosine search → decision
"""

import os
import streamlit as st
import cv2
import numpy as np
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# ── Project root ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.detection import SCRFDDetector
from src.alignment import FaceAligner
from src.recognition import FaceEmbedder
from src.quality import FaceQualityEstimator
from src.gallery import FaceGallery
from src.tracking import FaceTracker
from src.aggregation import FeatureAggregator
from src.open_set import OpenSetDecisionEngine
from src.demo_workflow import process_image, recognize, SCORE_THRESHOLD, MARGIN_THRESHOLD

# ── Page configuration ───────────────────────────────────────
st.set_page_config(
    page_title="DynaCampus-FR Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    /* Header accent bar */
    .header-bar {
        background: linear-gradient(90deg, #00d4aa 0%, #0099ff 100%);
        height: 4px;
        border-radius: 2px;
        margin-bottom: 1.5rem;
    }
    /* Status badges */
    .badge-known {
        background: #00d4aa; color: #000; padding: 4px 14px;
        border-radius: 20px; font-weight: 700; font-size: 0.85rem;
        display: inline-block;
    }
    .badge-unknown {
        background: #ff6b6b; color: #fff; padding: 4px 14px;
        border-radius: 20px; font-weight: 700; font-size: 0.85rem;
        display: inline-block;
    }
    .badge-quality {
        background: #3d5afe; color: #fff; padding: 3px 10px;
        border-radius: 12px; font-size: 0.78rem;
        display: inline-block;
    }
    /* Metric card styling */
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
    }
    /* Clean dividers */
    hr { border-color: #30363d !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING  (cached — loaded once per session)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_detector():
    """Load SCRFD face detector."""
    try:
        return SCRFDDetector(config_path=str(ROOT / "configs" / "models.yaml"))
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_aligner():
    """Load face aligner."""
    return FaceAligner(crop_size=112)


@st.cache_resource(show_spinner=False)
def load_embedder():
    """Load the real recognizer; model failures must remain visible."""
    return FaceEmbedder(config_path=str(ROOT / "configs" / "models.yaml"))


@st.cache_resource(show_spinner=False)
def load_quality():
    """Load face quality estimator."""
    return FaceQualityEstimator()


def get_gallery():
    """Get or create the face gallery (stored in session state)."""
    if "gallery" not in st.session_state or not hasattr(st.session_state.gallery, "list_pending"):
        st.session_state.gallery = FaceGallery(
            gallery_dir=os.environ.get("DYNACAMPUS_GALLERY_DIR", str(ROOT / "gallery_data")),
            similarity_threshold=0.40,
        )
    return st.session_state.gallery


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def load_uploaded_image(uploaded_file) -> np.ndarray:
    """Convert Streamlit UploadedFile to BGR numpy array."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    uploaded_file.seek(0)  # Reset for potential re-read
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def process_faces(image, detector, aligner, embedder, quality_est, face_crop=False):
    """Use the real-model still-image path matching the evaluated protocol."""
    if detector is None:
        return []
    return process_image(image, detector, aligner, embedder, quality_est, face_crop)


def annotate_frame(image, processed_faces, gallery, threshold):
    """
    Draw bounding boxes and recognition labels on image.

    Returns:
        annotated_image, list_of_result_dicts
    """
    annotated = image.copy()
    results = []

    for face in processed_faces:
        result = recognize(gallery, face["embedding"], face["quality"]["overall_quality"],
                           threshold, margin_threshold, quality_threshold)
        bbox = face["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if result["decision"] == "KNOWN":
            color = (0, 200, 100)  # Green (BGR)
            label = f"{result['top1_name']} ({result['top1_score']:.2f})"
        elif result["decision"] == "AMBIGUOUS":
            color, label = (0, 220, 255), "AMBIGUOUS"
        elif result["decision"] == "INSUFFICIENT_QUALITY":
            color, label = (150, 150, 150), "LOW QUALITY"
        else:
            color = (80, 80, 255)
            label = f"UNKNOWN ({result['top1_score']:.2f})"

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(
            annotated, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, y1 - 6),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        # Quality badge (small text at bottom of bbox)
        q = face["quality"]["overall_quality"]
        q_label = f"Q:{q:.2f}"
        cv2.putText(
            annotated,
            q_label,
            (x1 + 2, y2 - 4),
            font,
            0.40,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        results.append(
            {
                "Decision": result["decision"],
                "Name": result.get("accepted_name") or "—",
                "Score": result.get("top1_score", 0.0),
                "Margin": result.get("margin", 0.0),
                "Quality": q,
                "Det. Conf.": round(face["score"], 3),
            }
        )

    return annotated, results


def add_info_bar(frame, gallery_size, fps=None):
    """Add a small info bar at the bottom of a frame."""
    h, w = frame.shape[:2]
    bar_h = 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (13, 17, 23), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    text = f"DynaCampus-FR  |  Gallery: {gallery_size}"
    if fps is not None:
        text += f"  |  FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        (8, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 212, 170),
        1,
        cv2.LINE_AA,
    )
    return frame


def bgr_to_rgb(image):
    """Convert BGR image to RGB for Streamlit display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 DynaCampus-FR")
    st.caption("ArcFace Open-Set\nFace Recognition Dashboard")
    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "📊 Dashboard",
            "➕ Enroll Person",
            "🖼️ Image Recognition",
            "🎬 Video Recognition",
            "📹 Live Webcam",
            "📡 Live CCTV (RTSP)",
            "⏳ Pending Approvals",
            "📋 Gallery Management",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    preset = st.radio("Decision settings", ["Frozen research settings", "Manual demo settings"])
    if preset == "Frozen research settings":
        threshold, margin_threshold, quality_threshold = SCORE_THRESHOLD, MARGIN_THRESHOLD, 0.0
        st.caption("Score 0.3779 · margin 0.15 · quality filtering off")
    else:
        threshold = st.slider("Recognition threshold", 0.10, 0.90, 0.40, 0.01)
        margin_threshold = st.slider("Minimum top-two margin", 0.0, 0.5, 0.15, 0.01)
        quality_threshold = st.slider("Minimum quality", 0.0, 1.0, 0.0, 0.05)
        st.caption("Manual settings are for demonstration; reported test settings stay frozen.")
    face_crop_input = st.checkbox("Input is a tight face crop", value=False,
        help="For enrollment and image uploads already cropped around one face.")
    show_previews = st.checkbox("Show photo previews", value=False)
    auto_capture = st.checkbox("Queue unknown faces for review", value=False)

    st.divider()
    gallery = get_gallery()
    gallery_counter = st.empty()
    gallery_counter.metric("Enrolled Identities", gallery.size)


# ═══════════════════════════════════════════════════════════════
#  LOAD MODELS  (with spinner on first load)
# ═══════════════════════════════════════════════════════════════
with st.spinner("⏳ Loading face detection and ArcFace recognition models…"):
    detector = load_detector()
    aligner = load_aligner()
    try:
        embedder = load_embedder()
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        st.error(f"Recognition model unavailable: {error}")
        st.stop()
    quality_est = load_quality()

if detector is None:
    st.error(
        "❌ **Failed to load the SCRFD face detector.**  \n"
        "Make sure `insightface` and `onnxruntime` are installed and you have internet access for the first model download."
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════
if "📊 Dashboard" in page:
    st.markdown("# 📊 Dashboard")
    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👤 Enrolled", gallery.size)
    c2.metric("🧠 Backbone", "ArcFace")
    c3.metric("🔍 Detector", "SCRFD")
    c4.metric("📏 Embedding", "512-d")

    st.divider()

    if gallery.size > 0:
        st.subheader("Enrolled Faces")
        identities = gallery.list_identities()
        cols = st.columns(min(gallery.size, 6))
        for i, identity in enumerate(identities):
            with cols[i % len(cols)]:
                thumb = gallery.get_thumbnail(identity["subject_id"])
                if thumb is not None:
                    if show_previews:
                        st.image(bgr_to_rgb(thumb), caption=identity["name"], width=120)
                else:
                    st.info(identity["name"])
    else:
        st.info(
            "🚀 **Gallery is empty.** Go to **➕ Enroll Person** to add identities."
        )

    st.divider()
    st.subheader("System Architecture")
    st.markdown(
        """
        ```
        Enrollment:  Photo → SCRFD → Align → Embed → Quality → Gallery INSERT
        Recognition: Image/Video → SCRFD → Align → Embed → Gallery Search → KNOWN / Not Recognized
        ```
        **No model retraining** occurs when a new person is enrolled — only a template insertion.
        """
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE: ENROLL PERSON
# ═══════════════════════════════════════════════════════════════
elif "➕ Enroll" in page:
    st.markdown("# ➕ Enroll New Person")
    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)
    st.caption(
        "Upload one or more photos to create an identity template. "
        "The recognition backbone is **never retrained** — the gallery is simply updated."
    )

    col_form, col_preview = st.columns([1, 2])

    with col_form:
        name = st.text_input("Full Name *", placeholder="e.g. Raisul Islam")
        subject_id = st.text_input(
            "Subject ID (auto if blank)",
            placeholder=gallery.next_id(),
        )
        if not subject_id.strip():
            subject_id = gallery.next_id()

        uploaded_files = st.file_uploader(
            "Upload Photo(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

    if uploaded_files and len(uploaded_files) > 3:
        st.error("Choose up to three original enrollment photos.")
        st.stop()

    if uploaded_files and name.strip():
        all_embeddings = []
        all_qualities = []
        best_aligned = None
        best_quality = -1.0

        with col_preview:
            for idx, uploaded in enumerate(uploaded_files):
                image = load_uploaded_image(uploaded)
                if image is None:
                    st.warning(f"Could not read {uploaded.name}")
                    continue

                faces = process_faces(image, detector, aligner, embedder, quality_est, face_crop=face_crop_input)
                if not faces:
                    st.warning(f"⚠️ No face detected in **{uploaded.name}**")
                    continue

                if len(faces) != 1:
                    st.warning("Use an enrollment photo containing exactly one face.")
                    continue
                face = faces[0]

                all_embeddings.append(face["embedding"])
                all_qualities.append(face["quality"]["overall_quality"])

                if face["quality"]["overall_quality"] > best_quality:
                    best_quality = face["quality"]["overall_quality"]
                    best_aligned = face["aligned"]

                # Show preview
                c1, c2 = st.columns(2)
                with c1:
                    disp = image.copy()
                    bx = face["bbox"]
                    cv2.rectangle(disp, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 200, 100), 2)
                    if show_previews:
                        st.image(bgr_to_rgb(disp), caption=f"Detected — {uploaded.name}", use_container_width=True)
                with c2:
                    if show_previews:
                        st.image(bgr_to_rgb(face["aligned"]), caption="Aligned 112×112 Crop", width=150)
                    st.markdown(
                        f"**Clean Quality:** `{face['quality']['overall_quality']:.3f}`  |  "
                        f"Blur: `{face['quality']['blur_score']:.2f}`  "
                        f"Res: `{face['quality']['resolution_score']:.2f}`"
                    )


        if all_embeddings:
            st.divider()
            st.info(
                f"✅ **{len(all_embeddings)}** original-photo feature vector(s) ready.  "
                f"Mean image quality: **{np.mean(all_qualities):.3f}**"
            )

            if st.button("✅ Confirm Enrollment", type="primary", use_container_width=True):
                success = gallery.enroll_multi(
                    subject_id,
                    name.strip(),
                    all_embeddings,
                    all_qualities,
                    best_aligned,
                )

                if success:
                    # Force gallery reload in session state
                    st.session_state.gallery = gallery
                    gallery_counter.metric("Enrolled Identities", gallery.size)
                    st.success(
                        f"🎉 **{name.strip()}** enrolled as **{subject_id}**!  "
                        f"Gallery now has **{gallery.size}** identities.  \n"
                        f"⚡ **No model retraining occurred** — template inserted directly."
                    )
                    st.balloons()
                else:
                    st.error("Enrollment failed. Check image quality.")
    elif uploaded_files and not name.strip():
        st.warning("Please enter a **name** to proceed with enrollment.")


# ═══════════════════════════════════════════════════════════════
#  PAGE: IMAGE RECOGNITION
# ═══════════════════════════════════════════════════════════════
elif "🖼️ Image" in page:
    st.markdown("# 🖼️ Image Recognition")
    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)

    if gallery.size == 0:
        st.warning("⚠️ Gallery is empty. Enroll at least one person first.")
    else:
        st.caption(
            f"Upload an image to detect and recognize faces against **{gallery.size}** enrolled identities."
        )

    uploaded = st.file_uploader(
        "Upload Image", type=["jpg", "jpeg", "png"], key="img_rec"
    )

    if uploaded:
        image = load_uploaded_image(uploaded)
        if image is None:
            st.error("Failed to read image.")
        else:
            with st.spinner("Detecting and recognizing faces…"):
                faces = process_faces(image, detector, aligner, embedder, quality_est, face_crop=face_crop_input)

            if not faces:
                st.warning("No faces detected in this image.")
                if show_previews:
                    st.image(bgr_to_rgb(image), caption="Original Image", use_container_width=True)
            else:
                annotated, results = annotate_frame(image, faces, gallery, threshold)
                annotated = add_info_bar(annotated, gallery.size)

                # Display
                if show_previews:
                    st.image(
                        bgr_to_rgb(annotated),
                        caption=f"{len(faces)} face(s) detected",
                        use_container_width=True,
                    )

                # Results table
                st.divider()
                st.subheader("Recognition Results")

                for i, r in enumerate(results):
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        if show_previews:
                            st.image(bgr_to_rgb(faces[i]["aligned"]), width=90)
                    with c2:
                        if r["Decision"] == "KNOWN":
                            st.markdown(
                                f'<span class="badge-known">✅ KNOWN</span> &nbsp; **{r["Name"]}**',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<span class="badge-unknown">{r["Decision"]}</span>',
                                unsafe_allow_html=True,
                            )
                        st.caption(
                            f"Score: **{r['Score']:.4f}** &nbsp;|&nbsp; "
                            f"Margin: {r['Margin']:.4f} &nbsp;|&nbsp; "
                            f"Quality: {r['Quality']:.3f} &nbsp;|&nbsp; "
                            f"Det. Confidence: {r['Det. Conf.']:.3f}"
                        )

                st.divider()
                st.dataframe(pd.DataFrame(results), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: VIDEO RECOGNITION
# ═══════════════════════════════════════════════════════════════
elif "🎬 Video" in page:
    st.markdown("# 🎬 Video Recognition")
    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)

    if gallery.size == 0:
        st.warning("⚠️ Gallery is empty. Enroll at least one person first.")

    st.caption("Upload a video file to process frame-by-frame with face recognition.")

    uploaded_video = st.file_uploader(
        "Upload Video", type=["mp4", "avi", "mov", "mkv"], key="vid_rec"
    )

    if uploaded_video:
        # Save to temp file for OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.info(
            f"📹 **{uploaded_video.name}** — {total_frames} frames, "
            f"{fps_video:.1f} FPS, {frame_w}×{frame_h}"
        )

        process_every = st.slider(
            "Process every N-th frame (↑ = faster, ↓ = more detailed)",
            min_value=1,
            max_value=10,
            value=3,
        )

        if st.button("▶️ Process Video", type="primary"):
            frame_placeholder = st.empty()
            progress_bar = st.progress(0, text="Processing…")
            all_results = []
            frame_idx = 0
            t_start = time.time()

            # Prepare video writer for annotated output
            out_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ).name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = fps_video / process_every
            writer = cv2.VideoWriter(out_path, fourcc, out_fps, (frame_w, frame_h))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % process_every == 0:
                    faces = process_faces(
                        frame, detector, aligner, embedder, quality_est
                    )
                    annotated, results = annotate_frame(
                        frame, faces, gallery, threshold
                    )
                    annotated = add_info_bar(annotated, gallery.size)

                    for r in results:
                        r["Frame"] = frame_idx
                    all_results.extend(results)

                    writer.write(annotated)
                    frame_placeholder.image(
                        bgr_to_rgb(annotated), use_container_width=True
                    )

                frame_idx += 1
                if total_frames > 0:
                    progress_bar.progress(
                        min(frame_idx / total_frames, 1.0),
                        text=f"Frame {frame_idx}/{total_frames}",
                    )

            cap.release()
            writer.release()
            elapsed = time.time() - t_start
            progress_bar.progress(1.0, text="✅ Complete!")

            # Summary
            st.divider()
            st.subheader("📊 Results Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Frames Processed", frame_idx // process_every)
            c2.metric("Faces Detected", len(all_results))
            known_count = sum(1 for r in all_results if r["Decision"] == "KNOWN")
            c3.metric("Known", known_count)
            c4.metric("Processing Time", f"{elapsed:.1f}s")

            if all_results:
                st.divider()
                st.subheader("Detailed Recognition Log")
                df = pd.DataFrame(all_results)
                st.dataframe(df, use_container_width=True)

            # Download annotated video
            if Path(out_path).exists() and Path(out_path).stat().st_size > 0:
                st.divider()
                with open(out_path, "rb") as vf:
                    st.download_button(
                        "⬇️ Download Annotated Video",
                        data=vf,
                        file_name="annotated_output.mp4",
                        mime="video/mp4",
                    )

        cap.release()

        # Cleanup temp file
        try:
            Path(tfile.name).unlink(missing_ok=True)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
#  PAGE: LIVE WEBCAM
# ═══════════════════════════════════════════════════════════════
elif "📹 Live" in page:
    st.markdown("# 📹 Live Webcam Recognition")
    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)

    if gallery.size == 0:
        st.warning(
            "⚠️ Gallery is empty. Enroll at least one person first for recognition."
        )

    st.caption(
        "Start the webcam to run real-time face detection and recognition. "
        "Check the box below to begin."
    )

    cam_index = st.selectbox("Camera Index", [0, 1, 2], index=0)

    run_webcam = st.checkbox("🎥 Start Live Recognition", value=False)

    if run_webcam:
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        log_placeholder = st.empty()

        cap = cv2.VideoCapture(int(cam_index))

        if not cap.isOpened():
            st.error(
                "❌ Cannot access webcam. Make sure it's connected and not in use by another application."
            )
        else:
            st.info("📹 Webcam running — **uncheck the box above** to stop.")
            recent_log = []
            frame_count = 0
            t_start = time.time()

            try:
                tracker = FaceTracker(max_age=15, min_hits=2, iou_threshold=0.3)
                aggregator = FeatureAggregator(strategy="quality_weighted")
                osde = OpenSetDecisionEngine(base_threshold=threshold, quality_threshold=quality_threshold, margin_threshold=margin_threshold)
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Lost webcam connection.")
                        break

                    frame_count += 1
                    elapsed = time.time() - t_start
                    fps = frame_count / elapsed if elapsed > 0 else 0

                    # Process faces (raw detection + embedding)
                    faces = process_faces(
                        frame, detector, aligner, embedder, quality_est
                    )
                    
                    # 1. Prepare detections for tracker [x1, y1, x2, y2, score]
                    dets = []
                    for f in faces:
                        dets.append([f["bbox"][0], f["bbox"][1], f["bbox"][2], f["bbox"][3], f["score"]])
                    
                    if len(dets) > 0:
                        dets = np.array(dets)
                        tracks = tracker.update(dets)
                    else:
                        tracks = tracker.update(np.empty((0, 5)))
                    
                    annotated = frame.copy()
                    results = []
                    now = datetime.now().strftime("%H:%M:%S")
                    
                    # 2. Map tracks back to embeddings & aggregate
                    for trk in tracks:
                        x1, y1, x2, y2, track_id = trk
                        track_id = int(track_id)
                        
                        # Find closest face in current frame to this track (simplistic mapping for demo)
                        best_iou = 0
                        best_face = None
                        for f in faces:
                            # calculate simplistic center distance or IoU
                            cx1, cy1 = (x1+x2)/2, (y1+y2)/2
                            cx2, cy2 = (f["bbox"][0]+f["bbox"][2])/2, (f["bbox"][1]+f["bbox"][3])/2
                            dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                            if dist < 50: # pixel threshold
                                best_face = f
                                break
                                
                        if best_face:
                            # Update Aggregator
                            q = best_face["quality"]["overall_quality"]
                            aggregator.update(track_id, best_face["embedding"], q)
                            
                        # 3. Get aggregated feature and perform Open-Set Decision
                        agg_feat = aggregator.aggregate(track_id)
                        
                        label = f"ID: {track_id} (Tracking)"
                        color = (255, 150, 0) # Orange default
                        
                        if agg_feat is not None:
                            search_res = gallery.search(agg_feat, threshold=0.1) # search raw top matches
                            decision, match_id, match_score = osde.evaluate_scores(
                                search_res["top1_id"], search_res["top1_score"],
                                search_res["top2_score"] if gallery.size > 1 else None,
                                q if best_face else 0.5,
                            )
                            
                            if decision == "KNOWN":
                                color = (0, 200, 100) # Green
                                name = gallery.get_name(match_id)
                                label = f"{name} ({match_score:.2f})"
                                results.append({"Time": now, "Decision": "KNOWN", "Name": name, "Score": match_score})
                                
                                # --- Auto-Enrollment Prototype Logic ---
                                # If they were auto-enrolled previously, we could notify here.
                                
                            elif decision == "UNKNOWN":
                                color = (80, 80, 255) # Red
                                label = f"UNKNOWN ({match_score:.2f})"
                                results.append({"Time": now, "Decision": "UNKNOWN", "Name": "-", "Score": match_score})
                                
                                # Auto-Enrollment trigger
                                if auto_capture and best_face and q > 0.70:
                                    # High quality unknown face detected continuously!
                                    # Auto-enroll logic
                                    new_id = f"AutoID_{track_id}"
                                    if new_id not in gallery._identities:
                                        gallery.enroll(new_id, new_id, agg_feat, best_face["aligned"], q)
                                        results.append({"Time": now, "Decision": "AUTO_ENROLLED", "Name": new_id, "Score": q})
                                        
                            elif decision == "AMBIGUOUS":
                                color = (0, 255, 255) # Yellow
                                label = f"AMBIGUOUS ({match_score:.2f})"
                            elif decision == "INSUFFICIENT_QUALITY":
                                color = (150, 150, 150) # Gray
                                label = "LOW QUALITY"

                        # Draw Tracking BBox with crisp anti-aliased rendering
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
                        cv2.putText(annotated, label, (int(x1), max(int(y1)-8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                        
                    annotated = add_info_bar(annotated, gallery.size, fps=fps)

                    # Display annotated frame
                    frame_placeholder.image(
                        bgr_to_rgb(annotated), use_container_width=True
                    )

                    # Live stats (throttled to keep stream smooth at 30+ FPS)
                    if results:
                        for r in results:
                            r["Time"] = now
                            recent_log.append(r)
                        recent_log = recent_log[-20:]
                    
                    if frame_count % 15 == 0 and recent_log:
                        log_placeholder.dataframe(
                            pd.DataFrame(recent_log[::-1]),
                            use_container_width=True,
                        )

            except Exception as e:
                import traceback
                st.error(f"Error in webcam loop: {e}")
                st.code(traceback.format_exc())
            finally:
                cap.release()



# ═══════════════════════════════════════════════════════════════
#  PAGE: LIVE CCTV (RTSP STREAM)
# ═══════════════════════════════════════════════════════════════
elif "📡 Live CCTV" in page:
    st.markdown("# 📡 Live CCTV Stream (RTSP / IP Camera)")
    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)

    st.caption(
        "Connect to a campus CCTV feed via RTSP URL (e.g. `rtsp://admin:password@192.168.1.100:554/stream`). "
        "Applies multi-person SORT tracking, Quality-Aware Temporal Aggregation (QATA), and Open-Set Recognition."
    )

    rtsp_url = st.text_input("RTSP Stream URL", value="rtsp://127.0.0.1:8554/live", help="Enter IP camera RTSP endpoint.")
    run_cctv = st.checkbox("▶️ Connect to CCTV Stream", value=False)

    if run_cctv:
        frame_placeholder = st.empty()
        log_placeholder = st.empty()
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            st.error(f"❌ Failed to connect to RTSP stream: `{rtsp_url}`. Please verify network address & credentials.")
        else:
            st.success(f"Connected to RTSP stream: `{rtsp_url}`")
            tracker = FaceTracker(max_age=15, min_hits=2, iou_threshold=0.3)
            aggregator = FeatureAggregator(strategy="quality_weighted")
            osde = OpenSetDecisionEngine(base_threshold=threshold, quality_threshold=quality_threshold, margin_threshold=margin_threshold)
            recent_log = []
            frame_count = 0
            t_start = time.time()

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("RTSP stream disconnected or end of stream.")
                        break

                    frame_count += 1
                    elapsed = time.time() - t_start
                    fps = frame_count / elapsed if elapsed > 0 else 0

                    faces = process_faces(frame, detector, aligner, embedder, quality_est)
                    dets = [[f["bbox"][0], f["bbox"][1], f["bbox"][2], f["bbox"][3], f["score"]] for f in faces]
                    tracks = tracker.update(np.array(dets) if dets else np.empty((0, 5)))

                    annotated = frame.copy()
                    results = []
                    now = datetime.now().strftime("%H:%M:%S")

                    for trk in tracks:
                        x1, y1, x2, y2, track_id = trk
                        track_id = int(track_id)
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        best_face = None
                        for f in faces:
                            cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
                            cx2, cy2 = (f["bbox"][0] + f["bbox"][2]) / 2, (f["bbox"][1] + f["bbox"][3]) / 2
                            if np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) < 60:
                                best_face = f
                                break

                        q = best_face["quality"]["overall_quality"] if best_face else 0.5
                        if best_face:
                            aggregator.update(track_id, best_face["embedding"], q)

                        agg_feat = aggregator.aggregate(track_id)
                        label = f"Cam Track #{track_id}"
                        color = (255, 150, 0)

                        if agg_feat is not None:
                            search_res = gallery.search(agg_feat, threshold=0.1)
                            decision, match_id, match_score = osde.evaluate_scores(
                                search_res["top1_id"], search_res["top1_score"],
                                search_res["top2_score"] if gallery.size > 1 else None, q,
                            )

                            if decision == "KNOWN":
                                color = (0, 200, 100)
                                name = gallery.get_name(match_id)
                                label = f"{name} ({match_score:.2f})"
                                results.append({"Time": now, "Cam": "CCTV-1", "Decision": "KNOWN", "Name": name, "Score": match_score})
                            elif decision == "UNKNOWN":
                                color = (80, 80, 255)
                                label = f"UNKNOWN ({match_score:.2f})"
                                results.append({"Time": now, "Cam": "CCTV-1", "Decision": "UNKNOWN", "Name": "-", "Score": match_score})

                                if auto_capture and best_face and q > 0.70:
                                    auto_id = f"AutoID_{track_id}"
                                    if auto_id not in gallery._identities:
                                        gallery.enroll(auto_id, auto_id, agg_feat, best_face["aligned"], q)
                            elif decision == "AMBIGUOUS":
                                color = (0, 255, 255)
                                label = f"AMBIGUOUS ({match_score:.2f})"
                                results.append({"Time": now, "Cam": "CCTV-1", "Decision": decision, "Name": "-", "Score": match_score})
                            elif decision == "INSUFFICIENT_QUALITY":
                                color = (150, 150, 150)
                                label = "LOW QUALITY"
                                results.append({"Time": now, "Cam": "CCTV-1", "Decision": decision, "Name": "-", "Score": match_score})

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                        cv2.putText(annotated, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

                    annotated = add_info_bar(annotated, gallery.size, fps=fps)
                    frame_placeholder.image(bgr_to_rgb(annotated), use_container_width=True)

                    if results:
                        recent_log.extend(results)
                        recent_log = recent_log[-20:]

                    if frame_count % 15 == 0 and recent_log:
                        log_placeholder.dataframe(pd.DataFrame(recent_log[::-1]), use_container_width=True)

            except Exception as e:
                st.error(f"CCTV stream error: {e}")
            finally:
                cap.release()


# ═══════════════════════════════════════════════════════════════
#  PAGE: PENDING APPROVALS
# ═══════════════════════════════════════════════════════════════
elif "⏳ Pending" in page:
    st.markdown("# ⏳ Pending Auto-Enrollments")
    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)

    st.caption(
        "Review high-quality unknown faces automatically captured by the CCTV & webcam feeds. "
        "Type a name and click **Approve** to permanently enroll them into the campus gallery."
    )

    pending_list = gallery.list_pending()

    if not pending_list:
        st.info("🎉 No pending auto-enrollments! New high-quality unknown faces will automatically appear here.")
    else:
        st.caption(f"Found **{len(pending_list)}** auto-captured faces waiting for approval.")

        cols_per_row = 3
        for idx, item in enumerate(pending_list):
            sid = item["subject_id"]
            if idx % cols_per_row == 0:
                cols = st.columns(cols_per_row)

            col = cols[idx % cols_per_row]
            with col:
                st.markdown(f"### 👤 `{sid}`")
                thumb = gallery.get_thumbnail(sid)
                if thumb is not None:
                    if show_previews:
                        st.image(bgr_to_rgb(thumb), use_container_width=True)

                st.caption(f"Quality Score: `{item.get('quality_score', 0):.3f}`  \nCaptured: {item.get('enrolled_at', '')[:19]}")

                new_name = st.text_input("Enter Full Name", key=f"name_{sid}", placeholder="e.g. Raisul Hasan")
                btn_col1, btn_col2 = st.columns(2)

                with btn_col1:
                    if st.button("✅ Approve", key=f"app_{sid}", type="primary"):
                        if new_name.strip():
                            gallery.approve_pending(sid, new_name.strip())
                            st.session_state.gallery = gallery
                            gallery_counter.metric("Enrolled Identities", gallery.size)
                            st.success(f"Enrolled `{new_name.strip()}`!")
                            st.rerun()
                        else:
                            st.warning("Please type a name first.")

                with btn_col2:
                    if st.button("🗑️ Discard", key=f"disc_{sid}"):
                        gallery.remove(sid)
                        st.session_state.gallery = gallery
                        gallery_counter.metric("Enrolled Identities", gallery.size)
                        st.info("Discarded entry.")
                        st.rerun()


# ═══════════════════════════════════════════════════════════════
#  PAGE: GALLERY MANAGEMENT
# ═══════════════════════════════════════════════════════════════
elif "📋 Gallery" in page:
    st.markdown("# 📋 Gallery Management")
    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)

    identities = gallery.list_identities()

    if not identities:
        st.info("Gallery is empty. Go to **➕ Enroll Person** to add people.")
    else:
        st.caption(f"**{len(identities)}** identities enrolled.")

        # Display as grid
        cols_per_row = 4
        for row_start in range(0, len(identities), cols_per_row):
            row_items = identities[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for i, identity in enumerate(row_items):
                with cols[i]:
                    sid = identity["subject_id"]
                    thumb = gallery.get_thumbnail(sid)
                    if thumb is not None:
                        if show_previews:
                            st.image(bgr_to_rgb(thumb), use_container_width=True)
                    else:
                        st.markdown("👤")

                    st.markdown(f"**{identity['name']}**")
                    st.caption(
                        f"ID: `{sid}`  \n"
                        f"Quality: `{identity.get('quality_score', 0):.3f}`  \n"
                        f"Enrolled: {identity.get('enrolled_at', 'N/A')[:10]}"
                    )

                    if st.button(f"🗑️ Remove", key=f"del_{sid}"):
                        gallery.remove(sid)
                        st.session_state.gallery = gallery
                        gallery_counter.metric("Enrolled Identities", gallery.size)
                        st.rerun()

        st.divider()
        if st.button("🗑️ Clear Entire Gallery", type="secondary"):
            for identity in identities:
                gallery.remove(identity["subject_id"])
            st.session_state.gallery = gallery
            gallery_counter.metric("Enrolled Identities", gallery.size)
            st.success("Gallery cleared.")
            st.rerun()
