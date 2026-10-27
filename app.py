"""Streamlit dashboard for the Smart Attendance System."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pandas as pd
import streamlit as st

from src.attendance import AttendanceLogger
from src.config import ensure_project_dirs, settings
from src.encode import FaceEncodingStore
from src.utils import (
    bgr_to_rgb,
    draw_face_box,
    list_image_files,
    list_people,
    open_camera,
    overlay_timestamp,
    sanitize_person_name,
)


ensure_project_dirs()


st.set_page_config(
    page_title="Smart Attendance",
    page_icon="SA",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 1280px; }
    [data-testid="stSidebar"] { background: #101820; }
    [data-testid="stSidebar"] * { color: #f8fafc; }
    h1, h2, h3 { letter-spacing: 0; }
    .metric-row [data-testid="stMetric"] {
        background: #f7f9fb;
        border: 1px solid #dfe7ef;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.2rem 0.58rem;
        font-size: 0.82rem;
        font-weight: 700;
        background: #e8f3ff;
        color: #075985;
        border: 1px solid #bfdbfe;
    }
    .danger-pill {
        background: #fff1f2;
        color: #be123c;
        border-color: #fecdd3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_logs(storage: str, limit: int | None = 200) -> pd.DataFrame:
    logger = AttendanceLogger(storage=storage)  # type: ignore[arg-type]
    rows = logger.read_logs(limit=limit)
    return pd.DataFrame(rows)


def show_runtime_error(title: str, exc: Exception) -> None:
    st.error(title)
    st.code(str(exc))


def import_cv2():
    import cv2

    return cv2


def import_face_registrar():
    from src.register import FaceRegistrar

    return FaceRegistrar


def import_face_recognizer():
    from src.recognize import FaceRecognizer

    return FaceRecognizer


def decode_image_file(file_obj) -> np.ndarray | None:
    """Decode Streamlit uploaded or camera image to OpenCV BGR format."""
    if file_obj is None:
        return None
    try:
        cv2 = import_cv2()
        bytes_data = file_obj.getvalue()
        if not bytes_data:
            return None
        file_bytes = np.frombuffer(bytes_data, np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return frame_bgr
    except Exception as exc:  # noqa: BLE001
        st.error(f"Image decode error: {exc}")
        return None


def dashboard_page() -> None:
    st.title("Smart Attendance System")

    people = list_people()
    image_count = len(list_image_files(settings.dataset_dir))
    names, encodings, _ = FaceEncodingStore().load()
    logs_df = load_logs("csv", limit=None)

    today_count = 0
    if not logs_df.empty and "date" in logs_df.columns:
        today_count = int((logs_df["date"] == time.strftime("%Y-%m-%d")).sum())

    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registered People", len(people))
    col2.metric("Dataset Images", image_count)
    col3.metric("Cached Encodings", len(encodings))
    col4.metric("Today Marks", today_count)
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Recent Attendance")
        if logs_df.empty:
            st.info("No attendance logs yet.")
        else:
            st.dataframe(logs_df.head(10), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Encoding Cache")
        store = FaceEncodingStore()
        cache_current = store.cache_is_current()
        pill_class = "status-pill" if cache_current else "status-pill danger-pill"
        label = "Current" if cache_current else "Needs rebuild"
        st.markdown(f'<span class="{pill_class}">{label}</span>', unsafe_allow_html=True)
        st.write(f"Cache file: `{settings.encodings_file.name}`")
        if st.button("Rebuild Encodings", use_container_width=True):
            try:
                report = store.build(force=True)
            except Exception as exc:  # noqa: BLE001
                show_runtime_error("Encoding rebuild failed.", exc)
            else:
                st.success(
                    f"Encoded {report.encoded_faces} faces from {report.total_images} images."
                )


def register_page() -> None:
    st.title("Register User")

    capture_mode = st.radio(
        "Capture Method",
        [
            "📸 Browser Camera (Webcam in Browser)",
            "📁 Upload Images (Batch / Drag & Drop)",
            "🖥️ Local / Server Camera (OpenCV)",
        ],
        horizontal=True,
    )

    if capture_mode == "📸 Browser Camera (Webcam in Browser)":
        col1, col2 = st.columns([1.3, 0.7])
        person_name = col1.text_input("Name", placeholder="Example: Vansh Gupta")
        samples_target = col2.number_input("Target Samples", min_value=5, max_value=50, value=25, step=5)

        if not person_name.strip():
            st.info("👆 Enter the person's name above to activate the camera.")
            return

        try:
            safe_name = sanitize_person_name(person_name)
        except ValueError as exc:
            st.error(str(exc))
            return

        FaceRegistrar = import_face_registrar()
        registrar = FaceRegistrar()
        existing_count = registrar.count_samples(safe_name)

        col2.metric("Saved Samples", f"{existing_count}/{samples_target}")
        progress = min(1.0, existing_count / max(samples_target, 1))
        st.progress(progress)

        next_pose = registrar.next_prompt(existing_count)
        st.info(f"💡 Recommended pose for next photo: **{next_pose}**")

        camera_file = st.camera_input(f"Take photo for {safe_name}", key=f"cam_{safe_name}")

        if camera_file is not None:
            img_bytes = camera_file.getvalue()
            img_hash = hashlib.md5(img_bytes).hexdigest()
            last_hash_key = f"last_hash_{safe_name}"

            if st.session_state.get(last_hash_key) != img_hash:
                frame_bgr = decode_image_file(camera_file)
                if frame_bgr is not None:
                    result = registrar.validate_and_save_frame(frame_bgr, safe_name)
                    st.session_state[last_hash_key] = img_hash
                    st.session_state[f"last_res_{safe_name}"] = result
                    st.session_state[f"last_frame_{safe_name}"] = frame_bgr
                    if result.saved:
                        st.rerun()

            last_res = st.session_state.get(f"last_res_{safe_name}")
            last_frame = st.session_state.get(f"last_frame_{safe_name}")
            if last_res:
                if last_res.saved:
                    st.success(f"✅ {last_res.message} (Total: {registrar.count_samples(safe_name)}/{samples_target})")
                else:
                    st.warning(f"⚠️ {last_res.message}")
                if last_frame is not None and last_res.face_location:
                    annotated = last_frame.copy()
                    color = (40, 180, 80) if last_res.saved else (0, 180, 255)
                    draw_face_box(annotated, last_res.face_location, last_res.status.replace("_", " ").title(), color)
                    st.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)

        st.divider()
        col_btn1, col_btn2 = st.columns([1, 1])
        if col_btn1.button("⚡ Rebuild Encodings Now", type="primary", use_container_width=True):
            store = FaceEncodingStore()
            with st.spinner("Rebuilding face encodings..."):
                report = store.build(force=True)
                st.success(f"Trained model on {report.encoded_faces} faces across {report.people} people! Ready for attendance.")

        if col_btn2.button("🔄 Clear Camera Photo Buffer", use_container_width=True):
            st.session_state.pop(f"last_hash_{safe_name}", None)
            st.rerun()

    elif capture_mode == "📁 Upload Images (Batch / Drag & Drop)":
        col1, col2 = st.columns([1.3, 0.7])
        person_name = col1.text_input("Name", placeholder="Example: Vansh Gupta", key="upload_name")
        auto_rebuild = col2.checkbox("Auto-rebuild model after saving", value=True)

        uploaded_files = st.file_uploader(
            "Upload 10-30 Face Photos / Selfies (JPEG/PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Select multiple photos of the person from different angles for optimal recognition accuracy.",
        )

        if st.button("Process & Save Images", type="primary", use_container_width=True):
            if not person_name.strip():
                st.error("Please enter a person name before processing.")
                return
            if not uploaded_files:
                st.warning("Please upload at least one image file.")
                return

            try:
                safe_name = sanitize_person_name(person_name)
            except ValueError as exc:
                st.error(str(exc))
                return

            FaceRegistrar = import_face_registrar()
            registrar = FaceRegistrar()

            saved_count = 0
            skipped_count = 0
            progress_bar = st.progress(0.0)

            for idx, file_obj in enumerate(uploaded_files):
                frame_bgr = decode_image_file(file_obj)
                if frame_bgr is not None:
                    result = registrar.validate_and_save_frame(frame_bgr, safe_name)
                    if result.saved:
                        saved_count += 1
                    else:
                        skipped_count += 1
                progress_bar.progress((idx + 1) / len(uploaded_files))

            total_now = registrar.count_samples(safe_name)
            st.success(f"✅ Processed {len(uploaded_files)} images: Saved {saved_count} valid face samples! (Total: {total_now} samples for {safe_name}).")
            if skipped_count > 0:
                st.warning(f"⚠️ {skipped_count} images were skipped because no face or multiple faces were detected.")

            if auto_rebuild and saved_count > 0:
                store = FaceEncodingStore()
                with st.spinner("Training model with new face samples..."):
                    report = store.build(force=True)
                    st.success(f"⚡ Encodings rebuilt! Encoded {report.encoded_faces} faces across {report.people} people. You can now take attendance.")

    else:
        # Local / Server OpenCV Camera
        st.info("ℹ️ **Local Mode:** This mode directly accesses the webcam attached to the host machine via OpenCV. When running on Streamlit Cloud, please select **'📸 Browser Camera'** or **'📁 Upload Images'** above.")
        col1, col2, col3 = st.columns([1.4, 0.8, 0.8])
        person_name = col1.text_input("Name", placeholder="Example: Priya Sharma", key="local_person_name")
        samples = col2.slider(
            "Samples",
            settings.minimum_samples_per_user,
            settings.maximum_samples_per_user,
            settings.default_samples_per_user,
            key="local_samples",
        )
        camera_index = col3.number_input("Camera Index", min_value=0, max_value=5, value=settings.camera_index, key="local_cam_idx")

        frame_slot = st.empty()
        status_slot = st.empty()
        progress_slot = st.empty()

        if st.button("Capture Samples (Local OpenCV)", type="primary", use_container_width=True):
            try:
                safe_name = sanitize_person_name(person_name)
            except ValueError as exc:
                st.error(str(exc))
                return

            try:
                cv2 = import_cv2()
                FaceRegistrar = import_face_registrar()
                registrar = FaceRegistrar()
            except Exception as exc:  # noqa: BLE001
                show_runtime_error("Registration tools could not be loaded.", exc)
                return

            existing_count = registrar.count_samples(safe_name)
            target_count = existing_count + samples
            progress = progress_slot.progress(0.0)

            with st.spinner("Initializing camera..."):
                try:
                    camera = open_camera(int(camera_index))
                except RuntimeError as exc:
                    st.error(f"{exc}. If running on Streamlit Cloud, switch to '📸 Browser Camera' above.")
                    return

            last_capture = 0.0
            start_time = time.time()
            last_message = "Camera ready"

            current_count = existing_count
            try:
                while current_count < target_count:
                    ok, frame = camera.read()
                    if not ok:
                        st.error("Camera frame could not be read.")
                        break

                    now = time.time()
                    result = None
                    if now - last_capture >= settings.registration_capture_interval_sec:
                        result = registrar.validate_and_save_frame(frame, safe_name)
                        last_capture = now
                        last_message = result.message
                        if result and result.saved:
                            current_count += 1

                    annotated = frame.copy()
                    if result and result.face_location:
                        color = (40, 180, 80) if result.saved else (0, 180, 255)
                        draw_face_box(
                            annotated,
                            result.face_location,
                            result.status.replace("_", " ").title(),
                            color,
                        )

                    cv2.putText(
                        annotated,
                        f"{safe_name}: {current_count}/{target_count}",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated,
                        registrar.next_prompt(current_count),
                        (16, 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    frame_slot.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)
                    progress.progress(min(1.0, (current_count - existing_count) / samples))
                    status_slot.info(last_message)

                    if time.time() - start_time > settings.registration_timeout_sec:
                        st.warning("Registration timed out.")
                        break
                    time.sleep(0.01)
            finally:
                camera.release()

            final_count = registrar.count_samples(safe_name)
            st.success(f"{safe_name} now has {final_count} dataset images.")
            st.warning("Rebuild encodings before starting attendance.")


def attendance_page() -> None:
    st.title("Live Attendance")

    attendance_mode = st.radio(
        "Attendance Mode",
        [
            "📸 Browser Camera (Cloud & Mobile)",
            "📁 Upload Photo to Verify",
            "🖥️ Local / Server Camera (Continuous Stream)",
        ],
        horizontal=True,
    )

    if attendance_mode == "📸 Browser Camera (Cloud & Mobile)":
        col1, col2 = st.columns(2)
        storage = col1.selectbox("Storage", ["csv", "sqlite"], index=0, key="att_storage_browser")
        threshold = col2.slider("Distance Threshold", 40.0, 100.0, 85.0, step=1.0, key="att_thresh_browser", help="Lower is stricter, higher is more forgiving. Default is 85.0.")

        try:
            FaceRecognizer = import_face_recognizer()
            attendance_logger = AttendanceLogger(storage=storage)  # type: ignore[arg-type]
            recognizer = FaceRecognizer(
                attendance_logger=attendance_logger,
                recognition_threshold=threshold,
            )
            recognizer.lbph_threshold = float(threshold)
        except Exception as exc:  # noqa: BLE001
            show_runtime_error("Recognition tools could not be loaded.", exc)
            return

        if len(recognizer.label_to_name) == 0:
            st.warning("⚠️ No trained face model found. Please go to **Register User** to add faces and click 'Rebuild Encodings Now' first.")
            return

        st.caption(f"Registered users loaded: **{', '.join(sorted(set(recognizer.label_to_name.values())))}**")

        photo = st.camera_input("Scan Face for Attendance", key="browser_attendance_cam")
        if photo is not None:
            frame_bgr = decode_image_file(photo)
            if frame_bgr is not None:
                with st.spinner("Recognizing face..."):
                    annotated, results = recognizer.recognize_frame(
                        frame_bgr,
                        mark_attendance=True,
                        annotate=True,
                    )

                st.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)

                if results:
                    for item in results:
                        if item.classification == "Valid" and item.name != "Unknown":
                            if item.attendance_event and item.attendance_event.status == "marked":
                                st.success(f"🎉 **Attendance Marked!** Welcome, **{item.name}**! (Confidence: {item.confidence:.1f}%)")
                            else:
                                reason = item.attendance_event.reason if item.attendance_event else "Recent attendance already logged"
                                st.info(f"ℹ️ **{item.name}** recognized ({item.confidence:.1f}%). {reason}")
                        else:
                            st.error(f"❌ Unrecognized face (Distance: {item.distance:.1f} vs threshold {threshold:.1f}). Make sure you are registered and encodings are rebuilt.")

                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Name": item.name,
                                    "Confidence (%)": f"{item.confidence:.1f}%",
                                    "Class": item.classification,
                                    "Distance": f"{item.distance:.1f}" if item.distance is not None else "N/A",
                                    "Status": item.attendance_event.reason if item.attendance_event else "Logged",
                                }
                                for item in results
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.warning("⚠️ No face detected in the photo. Please center your face in good lighting and try again.")

    elif attendance_mode == "📁 Upload Photo to Verify":
        col1, col2 = st.columns(2)
        storage = col1.selectbox("Storage", ["csv", "sqlite"], index=0, key="att_storage_upload")
        threshold = col2.slider("Distance Threshold", 40.0, 100.0, 85.0, step=1.0, key="att_thresh_upload")

        try:
            FaceRecognizer = import_face_recognizer()
            attendance_logger = AttendanceLogger(storage=storage)  # type: ignore[arg-type]
            recognizer = FaceRecognizer(
                attendance_logger=attendance_logger,
                recognition_threshold=threshold,
            )
            recognizer.lbph_threshold = float(threshold)
        except Exception as exc:  # noqa: BLE001
            show_runtime_error("Recognition tools could not be loaded.", exc)
            return

        if len(recognizer.label_to_name) == 0:
            st.warning("⚠️ No trained face model found. Please go to **Register User** first.")
            return

        uploaded_photo = st.file_uploader("Upload Image to Mark Attendance", type=["jpg", "jpeg", "png"], key="upload_att_file")
        if uploaded_photo is not None:
            frame_bgr = decode_image_file(uploaded_photo)
            if frame_bgr is not None:
                with st.spinner("Processing image..."):
                    annotated, results = recognizer.recognize_frame(
                        frame_bgr,
                        mark_attendance=True,
                        annotate=True,
                    )
                st.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)
                if results:
                    for item in results:
                        if item.classification == "Valid" and item.name != "Unknown":
                            if item.attendance_event and item.attendance_event.status == "marked":
                                st.success(f"🎉 **Attendance Marked!** Welcome, **{item.name}**! (Confidence: {item.confidence:.1f}%)")
                            else:
                                reason = item.attendance_event.reason if item.attendance_event else "Recent attendance already logged"
                                st.info(f"ℹ️ **{item.name}**: {reason}")
                        else:
                            st.error(f"❌ Face unrecognized or confidence too low ({item.distance:.1f} vs {threshold:.1f}).")
                else:
                    st.warning("⚠️ No face detected in the uploaded photo.")

    else:
        st.info("ℹ️ **Local Mode:** Continuous OpenCV streaming requires running on a computer with a local webcam plugged in. On Streamlit Cloud, use **'📸 Browser Camera'** above.")
        col1, col2, col3, col4 = st.columns(4)
        camera_index = col1.number_input("Camera", min_value=0, max_value=5, value=settings.camera_index, key="local_att_cam")
        storage = col2.selectbox("Storage", ["csv", "sqlite"], index=0, key="local_att_storage")
        duration = col3.slider("Session Seconds", 15, 600, 120, step=15, key="local_att_duration")
        threshold = col4.slider("Threshold", 40.0, 100.0, 85.0, step=1.0, key="local_att_threshold")

        require_liveness = st.checkbox("Require Blink", value=False, key="local_att_blink")
        mark_probable = st.checkbox("Mark Probable Matches", value=False, key="local_att_prob")

        frame_slot = st.empty()
        result_slot = st.empty()
        metric_slot = st.empty()

        if st.button("Start Attendance Session (OpenCV)", type="primary", use_container_width=True):
            try:
                FaceRecognizer = import_face_recognizer()
                attendance_logger = AttendanceLogger(storage=storage)  # type: ignore[arg-type]
                recognizer = FaceRecognizer(
                    attendance_logger=attendance_logger,
                    recognition_threshold=threshold,
                    require_liveness=require_liveness,
                    mark_probable=mark_probable,
                )
                recognizer.lbph_threshold = float(threshold)
            except Exception as exc:  # noqa: BLE001
                show_runtime_error("Recognition tools could not be loaded.", exc)
                return

            if len(recognizer.label_to_name) == 0:
                st.error("No known face encodings found. Register users and rebuild encodings first.")
                return

            try:
                cv2 = import_cv2()
                camera = open_camera(int(camera_index))
            except RuntimeError as exc:
                st.error(f"{exc}. If running on Streamlit Cloud, switch to '📸 Browser Camera' above.")
                return

            start = time.time()
            frame_count = 0
            processed_count = 0
            last_results = []

            try:
                while time.time() - start < duration:
                    ok, frame = camera.read()
                    if not ok:
                        st.error("Camera frame could not be read.")
                        break

                    frame_count += 1
                    if frame_count % settings.process_every_n_frames == 0:
                        annotated, last_results = recognizer.recognize_frame(
                            frame,
                            mark_attendance=True,
                            annotate=True,
                        )
                        processed_count += 1
                    else:
                        annotated = frame.copy()
                        overlay_timestamp(annotated)

                    frame_slot.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)

                    elapsed = max(time.time() - start, 1e-6)
                    fps = frame_count / elapsed
                    metric_slot.metric("Live FPS", f"{fps:.1f}", f"processed {processed_count}")

                    if last_results:
                        result_slot.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "name": item.name,
                                        "confidence": item.confidence,
                                        "class": item.classification,
                                        "distance": item.distance,
                                        "attendance": item.attendance_event.reason
                                        if item.attendance_event
                                        else "",
                                    }
                                    for item in last_results
                                ]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                    time.sleep(0.01)
            finally:
                camera.release()

            st.success("Attendance session ended.")


def logs_page() -> None:
    st.title("Attendance Logs")

    col1, col2 = st.columns([0.35, 0.65])
    storage = col1.selectbox("Storage", ["csv", "sqlite"], index=0, key="logs_storage")
    limit = col1.number_input("Rows", min_value=10, max_value=5000, value=200, step=10)

    attendance_logger = AttendanceLogger(storage=storage)  # type: ignore[arg-type]
    rows = attendance_logger.read_logs(limit=int(limit))
    logs_df = pd.DataFrame(rows)

    if logs_df.empty:
        col2.info("No logs found.")
    else:
        col2.dataframe(logs_df, use_container_width=True, hide_index=True)

    export_path = attendance_logger.export_csv(settings.logs_dir / "attendance_export.csv")
    st.download_button(
        "Export CSV",
        data=export_path.read_bytes(),
        file_name="attendance_export.csv",
        mime="text/csv",
        use_container_width=True,
    )


def dataset_page() -> None:
    st.title("Dataset Health")

    store = FaceEncodingStore()
    deep = st.checkbox("Deep Validation", value=False)

    if st.button("Run Validation", use_container_width=True):
        try:
            health = store.validate_dataset(deep=deep)
        except Exception as exc:  # noqa: BLE001
            show_runtime_error("Dataset validation failed.", exc)
        else:
            st.json(health)

    if st.button("Rebuild Encodings Now", type="primary", use_container_width=True):
        try:
            report = store.build(force=True)
        except Exception as exc:  # noqa: BLE001
            show_runtime_error("Encoding rebuild failed.", exc)
        else:
            st.success(
                f"Encoded {report.encoded_faces}/{report.total_images} images for {report.people} people."
            )
            if report.errors:
                st.warning(f"{len(report.errors)} images were skipped.")
                st.write(report.errors[:25])

    people = list_people()
    table = [
        {
            "person": person,
            "images": len(list_image_files(settings.dataset_dir / person)),
        }
        for person in people
    ]
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)


def setup_page() -> None:
    st.title("Setup")
    st.code("python -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt", language="bash")
    st.code("streamlit run app.py", language="bash")
    st.code("python -m src.register \"Student Name\"\npython -m src.encode --force\npython -m src.recognize", language="bash")


def main() -> None:
    st.sidebar.title("Smart Attendance")
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Register User", "Live Attendance", "Logs", "Dataset Health", "Setup"],
    )

    if page == "Dashboard":
        dashboard_page()
    elif page == "Register User":
        register_page()
    elif page == "Live Attendance":
        attendance_page()
    elif page == "Logs":
        logs_page()
    elif page == "Dataset Health":
        dataset_page()
    else:
        setup_page()


if __name__ == "__main__":
    main()
