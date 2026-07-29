"""
DynaCampus-FR — Research Dashboard
===================================
Streamlit application for face enrollment, recognition, and live video inference.
Demonstrates training-free dynamic enrollment with the full pipeline:
  SCRFD detection → alignment → embedding → gallery cosine search → decision
"""

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
    """Load face embedder (falls back to mock if models unavailable)."""
    try:
        return FaceEmbedder(config_path=str(ROOT / "configs" / "models.yaml"))
    except Exception:
        return FaceEmbedder(mock=True)


@st.cache_resource(show_spinner=False)
def load_quality():
    """Load face quality estimator."""
    return FaceQualityEstimator()


def get_gallery() -> FaceGallery:
    """Get or create the face gallery (stored in session state)."""
    if "gallery" not in st.session_state:
        st.session_state.gallery = FaceGallery(
            gallery_dir=str(ROOT / "gallery_data"),
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


def process_faces(image, detector, aligner, embedder, quality_est):
    """
    Run full pipeline on an image: detect → align → embed → quality.

    Returns list of dicts, each with keys:
      bbox, score, kps, aligned, embedding, quality
    """
    if detector is None:
        return []
    faces = detector.detect(image)
    processed = []
    for face in faces:
        kps = face.get("kps")
        if kps is None:
            continue
        kps_arr = np.array(kps, dtype=np.float32) if not isinstance(kps, np.ndarray) else kps
        if kps_arr.shape != (5, 2):
            continue
        try:
            aligned = aligner.align(image, kps_arr)
            embedding = embedder.extract_embedding(aligned)
            quality = quality_est.estimate_quality(
                aligned, bbox=face["bbox"], landmarks=kps_arr
            )
            processed.append(
                {
                    "bbox": face["bbox"],
                    "score": face["score"],
                    "kps": kps_arr,
                    "aligned": aligned,
                    "embedding": embedding,
                    "quality": quality,
                }
            )
        except Exception:
            continue
    return processed


def annotate_frame(image, processed_faces, gallery, threshold):
    """
    Draw bounding boxes and recognition labels on image.

    Returns:
        annotated_image, list_of_result_dicts
    """
    annotated = image.copy()
    results = []

    for face in processed_faces:
        result = gallery.search(face["embedding"], threshold=threshold)
        bbox = face["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if result["decision"] == "KNOWN":
            color = (0, 200, 100)  # Green (BGR)
            label = f"{result['top1_name']} ({result['top1_score']:.2f})"
        else:
            color = (80, 80, 255)  # Red (BGR)
            label = f"Not Recognized ({result['top1_score']:.2f})"

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
                "Name": result.get("top1_name") or "—",
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
    st.caption("Domain-Adaptive Open-Set\nFace Recognition Dashboard")
    st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "📊 Dashboard",
            "➕ Enroll Person",
            "🖼️ Image Recognition",
            "🎬 Video Recognition",
            "📹 Live Webcam",
            "📋 Gallery Management",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    threshold = st.slider(
        "Recognition Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Minimum cosine similarity to classify a face as KNOWN.",
    )

    st.divider()
    gallery = get_gallery()
    st.metric("Enrolled Identities", gallery.size)


# ═══════════════════════════════════════════════════════════════
#  LOAD MODELS  (with spinner on first load)
# ═══════════════════════════════════════════════════════════════
with st.spinner("⏳ Loading face detection & recognition models (first run downloads ~300 MB)…"):
    detector = load_detector()
    aligner = load_aligner()
    embedder = load_embedder()
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
    c2.metric("🧠 Backbone", "ArcFace / AdaFace")
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

                faces = process_faces(image, detector, aligner, embedder, quality_est)
                if not faces:
                    st.warning(f"⚠️ No face detected in **{uploaded.name}**")
                    continue

                # Take the largest face (highest bbox area)
                face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))

                all_embeddings.append(face["embedding"])
                q = face["quality"]["overall_quality"]
                all_qualities.append(q)

                if q > best_quality:
                    best_quality = q
                    best_aligned = face["aligned"]

                # Show preview
                c1, c2 = st.columns(2)
                with c1:
                    # Draw bbox on original
                    disp = image.copy()
                    bx = face["bbox"]
                    cv2.rectangle(
                        disp,
                        (int(bx[0]), int(bx[1])),
                        (int(bx[2]), int(bx[3])),
                        (0, 200, 100),
                        2,
                    )
                    st.image(bgr_to_rgb(disp), caption=f"Detected — {uploaded.name}", use_container_width=True)
                with c2:
                    st.image(bgr_to_rgb(face["aligned"]), caption="Aligned 112×112", width=150)
                    st.markdown(
                        f"**Quality:** `{q:.3f}`  |  "
                        f"Blur: `{face['quality']['blur_score']:.2f}`  "
                        f"Res: `{face['quality']['resolution_score']:.2f}`  "
                        f"Pose: `{face['quality']['pose_score']:.2f}`"
                    )

        if all_embeddings:
            st.divider()
            st.info(
                f"✅ **{len(all_embeddings)}** face(s) processed from **{len(uploaded_files)}** image(s).  "
                f"Average quality: **{np.mean(all_qualities):.3f}**"
            )

            if st.button("✅ Confirm Enrollment", type="primary", use_container_width=True):
                if len(all_embeddings) == 1:
                    success = gallery.enroll(
                        subject_id,
                        name.strip(),
                        all_embeddings[0],
                        best_aligned,
                        all_qualities[0],
                    )
                else:
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
                faces = process_faces(image, detector, aligner, embedder, quality_est)

            if not faces:
                st.warning("No faces detected in this image.")
                st.image(bgr_to_rgb(image), caption="Original Image", use_container_width=True)
            else:
                annotated, results = annotate_frame(image, faces, gallery, threshold)
                annotated = add_info_bar(annotated, gallery.size)

                # Display
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
                        st.image(bgr_to_rgb(faces[i]["aligned"]), width=90)
                    with c2:
                        if r["Decision"] == "KNOWN":
                            st.markdown(
                                f'<span class="badge-known">✅ KNOWN</span> &nbsp; **{r["Name"]}**',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                '<span class="badge-unknown">❌ NOT RECOGNIZED</span>',
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
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Lost webcam connection.")
                        break

                    frame_count += 1
                    elapsed = time.time() - t_start
                    fps = frame_count / elapsed if elapsed > 0 else 0

                    # Process faces
                    faces = process_faces(
                        frame, detector, aligner, embedder, quality_est
                    )
                    annotated, results = annotate_frame(
                        frame, faces, gallery, threshold
                    )
                    annotated = add_info_bar(annotated, gallery.size, fps=fps)

                    # Display annotated frame
                    frame_placeholder.image(
                        bgr_to_rgb(annotated), use_container_width=True
                    )

                    # Live stats
                    if results:
                        now = datetime.now().strftime("%H:%M:%S")
                        for r in results:
                            r["Time"] = now
                            recent_log.append(r)
                        # Keep last 30 entries
                        recent_log = recent_log[-30:]

                        with stats_placeholder.container():
                            cols = st.columns(min(len(results), 4))
                            for i, r in enumerate(results):
                                with cols[i % len(cols)]:
                                    if r["Decision"] == "KNOWN":
                                        st.success(f"✅ {r['Name']} ({r['Score']:.2f})")
                                    else:
                                        st.error(f"❌ Not Recognized ({r['Score']:.2f})")

                        log_placeholder.dataframe(
                            pd.DataFrame(recent_log[-10:][::-1]),
                            use_container_width=True,
                        )

            except Exception:
                pass
            finally:
                cap.release()


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
                        st.rerun()

        st.divider()
        if st.button("🗑️ Clear Entire Gallery", type="secondary"):
            for identity in identities:
                gallery.remove(identity["subject_id"])
            st.session_state.gallery = gallery
            st.success("Gallery cleared.")
            st.rerun()
