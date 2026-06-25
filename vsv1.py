# merged_app_auto.py — All-in-one Attendance System (with Group & Course Face Attendance)
# Auto Attendance: RegNo or QR -> fetch -> live face match -> auto save
# Fixed: single tab camera conflict by using a shared frame grabber for attendance tabs.
# NEW: Course-wise Attendance (select a course -> only those faces are matched & marked)
#
# Face recognition backend: InsightFace (buffalo_sc / ONNX)
# Install:  pip install insightface onnxruntime opencv-python pillow pandas pyzbar qrcode openpyxl
# NO dlib / NO cmake required.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import cv2, sqlite3, os, qrcode, shutil, base64
import numpy as np
from datetime import datetime
import pandas as pd
from pyzbar.pyzbar import decode
from insightface.app import FaceAnalysis

# ---------- InsightFace global init ----------
# buffalo_sc  = small + fast (recommended for CPU desktops)
# buffalo_l   = larger + more accurate  (slower on CPU)
#
# Model is stored in  <script_folder>/models/  so it is downloaded ONCE
# and reused on every subsequent run — no internet needed after first launch.
_MODEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(_MODEL_ROOT, exist_ok=True)

FA = FaceAnalysis(name="buffalo_sc", root=_MODEL_ROOT, providers=["CPUExecutionProvider"])
FA.prepare(ctx_id=0, det_size=(640, 480))


def safe_b64decode(data: str) -> str:
    data = data.strip().replace("\n", "").replace(" ", "")
    missing = len(data) % 4
    if missing:
        data += "=" * (4 - missing)
    return base64.b64decode(data).decode("utf-8", errors="ignore")

THEMES = {
    "dark": {
        "bg": "#121212",
        "fg": "white",
        "entry_bg": "#101010",
        "entry_fg": "#00ffff",
        "card_bg": "#1f1f1f",
        "hover_bg": "#2e7d32",
        "card_fg": "white",
        "shadow": "#00e5ff",
        "shadow_opacity": 200,
        "shadow_blur": 18,
        "shadow_blur_hover": 30,
        "shadow_opacity_hover": 230,
        "tree_bg": "#1f1f1f",
        "tree_fg": "white",
        "tree_header_bg": "#191919",
        "tree_header_fg": "white",
        "combo_bg": "#2b2b2b",
        "combo_fg": "white",
    },
    "light": {
        "bg": "white",
        "fg": "black",
        "entry_bg": "#f0f0f0",
        "entry_fg": "black",
        "card_bg": "#e0e0e0",
        "hover_bg": "#a5d6a7",
        "card_fg": "black",
        "shadow": "black",
        "shadow_opacity": 10,
        "shadow_blur": 18,
        "shadow_blur_hover": 30,
        "shadow_opacity_hover": 230,
        "tree_bg": "white",
        "tree_fg": "black",
        "tree_header_bg": "#dcdcdc",
        "tree_header_fg": "black",
        "combo_bg": "white",
        "combo_fg": "black",
    }
}

current_theme = "light"
open_windows = {}


def make_shadow_image(width, height, radius=20, shadow_color="#000000", blur_radius=12, offset=(5, 5)):
    total_width = width + offset[0] + blur_radius * 2
    total_height = height + offset[1] + blur_radius * 2
    img = Image.new("RGBA", (total_width, total_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    rect_x0 = blur_radius + offset[0]
    rect_y0 = blur_radius + offset[1]
    rect_x1 = rect_x0 + width
    rect_y1 = rect_y0 + height
    draw.rounded_rectangle([rect_x0, rect_y0, rect_x1, rect_y1], radius=radius, fill=shadow_color)
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    return img


# ---------- CONFIG ----------
LOGO_FILE = "logo.png"
DB_FILE   = "students.db"
APP_TITLE = "Central University of Andhra Pradesh - Attendance System"
DEVELOPER_TEXT = safe_b64decode("RGV2ZWxvcGVkIGJ5IFNhdHlhIER1cmdhIFJhbw==")

# InsightFace uses cosine similarity: higher = more similar.
# 0.35 is a solid threshold; raise to 0.40 for stricter matching.
SIMILARITY_THRESHOLD = 0.35
REQ_CONSEC           = 5    # consecutive matching frames to confirm (single mode)
GROUP_REQ_CONSEC     = 3    # slightly lower for group/course mode
PROCESS_EVERY_N      = 2    # process 1-in-N frames for performance
CAM_WIDTH            = 640
CAM_HEIGHT           = 480


# ---------- DB ----------
def get_conn():
    return sqlite3.connect(DB_FILE)

def init_db():
    os.makedirs("photos", exist_ok=True)
    os.makedirs("qrcodes", exist_ok=True)
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reg_no TEXT UNIQUE,
                name TEXT,
                course TEXT,
                mobile TEXT,
                photo_path TEXT,
                qr_path TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                date TEXT,
                time TEXT,
                match_percentage REAL,
                UNIQUE(student_id, date) ON CONFLICT IGNORE
            )
            """
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_student_date ON attendance(student_id, date)"
        )
        conn.commit()

init_db()


# ---------- Face helpers (InsightFace) ----------

def face_similarity(known_encodings: list, probe_enc: np.ndarray) -> np.ndarray:
    """
    Returns cosine similarity between probe_enc and every embedding in known_encodings.
    Values are in [-1, 1]; a match threshold of ~0.35 works well for buffalo_sc.
    """
    if not known_encodings:
        return np.array([])
    known = np.array(known_encodings, dtype=np.float32)
    probe = np.array(probe_enc, dtype=np.float32)
    # L2-normalise
    known_norm = known / (np.linalg.norm(known, axis=1, keepdims=True) + 1e-9)
    probe_norm = probe / (np.linalg.norm(probe) + 1e-9)
    return known_norm @ probe_norm        # shape (N,)


def safe_face_encoding_from_file(path: str):
    """Load a stored photo and return its 512-d InsightFace embedding, or None."""
    if not path or not os.path.exists(path):
        return None
    try:
        img = cv2.imread(path)           # BGR, as InsightFace expects
        if img is None:
            return None
        faces = FA.get(img)
        return faces[0].embedding if faces else None
    except Exception:
        return None


# Map: student_id -> embedding (lazy cache for current session)
ENCODING_CACHE: dict = {}


def load_all_face_encodings():
    known_encodings, known_ids, known_labels = [], [], []
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT id, reg_no, name, photo_path FROM students ORDER BY reg_no", conn
        )
    for _, row in df.iterrows():
        sid  = int(row["id"])
        reg  = row["reg_no"]
        name = row["name"]
        path = row["photo_path"]
        enc  = ENCODING_CACHE.get(sid)
        if enc is None:
            enc = safe_face_encoding_from_file(path)
            if enc is not None:
                ENCODING_CACHE[sid] = enc
        if enc is not None:
            known_encodings.append(enc)
            known_ids.append(sid)
            known_labels.append(f"{reg} | {name}" if name else reg)
    return known_encodings, known_ids, known_labels


def load_course_face_encodings(course_name: str):
    """Load embeddings only for students of a specific course."""
    known_encodings, known_ids, known_labels = [], [], []
    if not course_name:
        return known_encodings, known_ids, known_labels
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT id, reg_no, name, photo_path FROM students WHERE course = ? ORDER BY reg_no",
            conn,
            params=(course_name,),
        )
    for _, row in df.iterrows():
        sid  = int(row["id"])
        reg  = row["reg_no"]
        name = row["name"]
        path = row["photo_path"]
        enc  = ENCODING_CACHE.get(sid)
        if enc is None:
            enc = safe_face_encoding_from_file(path)
            if enc is not None:
                ENCODING_CACHE[sid] = enc
        if enc is not None:
            known_encodings.append(enc)
            known_ids.append(sid)
            known_labels.append(f"{reg} | {name}" if name else reg)
    return known_encodings, known_ids, known_labels


# ---------- Styled helpers ----------
def neon_entry(parent, var=None, readonly=False, width=28, bg="white"):
    e = tk.Entry(
        parent,
        textvariable=var,
        width=width,
        insertbackground="#00ffff",
        relief="flat",
        highlightthickness=1,
        highlightbackground="#0099aa",
        highlightcolor="#00ffff",
        font=("Segoe UI", 12),
    )
    if readonly:
        e.config(state="readonly", readonlybackground=bg)
    e.pack(pady=4, fill="x")
    return e


def neon_button(parent, text, command, bg="#0aa", fg="white"):
    b = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        bd=0,
        padx=12,
        pady=8,
        activebackground="#14c4c4",
        activeforeground="white",
        font=("Segoe UI", 11, "bold"),
    )
    b.pack(pady=6, fill="x")
    return b


# ---------- Enrollment ----------
def open_enrollment():
    if "enrollment" in open_windows:
        win = open_windows["enrollment"]
        if win.winfo_exists():
            win.deiconify(); win.lift(); win.focus_force()
            return
        open_windows.pop("enrollment", None)

    win = tk.Toplevel(root)
    win.title("Enrollment")
    win.geometry("880x560")
    open_windows["enrollment"] = win

    def on_close():
        open_windows.pop("enrollment", None)
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)

    theme = THEMES[current_theme]
    win.configure(bg=theme["bg"])

    form = tk.Frame(win)
    form.pack(side="left", padx=20, pady=20, fill="y")
    tk.Label(form, text="Register Student", font=("Segoe UI", 18, "bold"),
             fg=theme["fg"], bg=theme["bg"]).pack(pady=(0, 10))

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT course FROM students")
        courses = [r[0] for r in cur.fetchall()]

    vars_map = {}
    for label in ["Reg No", "Name", "Course", "Mobile"]:
        tk.Label(form, text=f"{label}:", fg=theme["fg"], bg=theme["bg"],
                 font=("Segoe UI", 11)).pack(anchor="w")
        if label == "Course":
            ent = ttk.Combobox(form, font=("Segoe UI", 13), values=courses)
            ent.pack(fill="x", pady=5)
            ent.set("")
        else:
            ent = tk.Entry(form, font=("Segoe UI", 13), bg=theme["bg"],
                           fg=theme["fg"], relief="solid")
            ent.pack(fill="x", pady=5)
        vars_map[label] = ent

    video_frame = tk.Frame(win)
    video_frame.pack(side="right", padx=20, pady=20)
    lbl_video = tk.Label(video_frame)
    lbl_video.pack()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    def update_cam():
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((int(CAM_WIDTH * 0.6), int(CAM_HEIGHT * 0.6)))
            imgtk = ImageTk.PhotoImage(img)
            lbl_video.imgtk = imgtk
            lbl_video.configure(image=imgtk)
        lbl_video.after(20, update_cam)

    update_cam()

    def save_student():
        reg    = vars_map["Reg No"].get().strip()
        name   = vars_map["Name"].get().strip()
        course = vars_map["Course"].get().strip()
        mobile = vars_map["Mobile"].get().strip()
        if not all([reg, name, course, mobile]):
            return messagebox.showerror("Error", "All fields required")

        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM students WHERE reg_no=?", (reg,))
            if cur.fetchone():
                return messagebox.showerror("Error", "Student already registered")

            ok, frame = cap.read()
            if not ok:
                return messagebox.showerror("Error", "Camera error")

            photo_path = os.path.join("photos", f"{reg}.jpg")
            cv2.imwrite(photo_path, frame)

            qr_path = os.path.join("qrcodes", f"{reg}.png")
            qrcode.make(reg).save(qr_path)

            cur.execute(
                "INSERT INTO students (reg_no, name, course, mobile, photo_path) VALUES (?, ?, ?, ?, ?)",
                (reg, name, course, mobile, photo_path),
            )
            conn.commit()

        # Clear cache entry if it exists
        try:
            with get_conn() as conn:
                sid = pd.read_sql_query(
                    "SELECT id FROM students WHERE reg_no = ?", conn, params=(reg,)
                )["id"].iloc[0]
                ENCODING_CACHE.pop(int(sid), None)
        except Exception:
            pass

        messagebox.showinfo("Success", f"{name} enrolled")
        for e in vars_map.values():
            e.delete(0, tk.END)

    neon_button(form, "📷 Capture & Save", save_student, bg="#4CAF50")

    def on_close():
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)


# ---------- Attendance: SINGLE + GROUP + COURSE (shared camera frame grabber) ----------
def open_attendance():
    if "attendance" in open_windows:
        win = open_windows["attendance"]
        if win.winfo_exists():
            win.deiconify(); win.lift(); win.focus_force()
            return
        open_windows.pop("attendance", None)

    win = tk.Toplevel(root)
    win.title("Attendance")
    win.geometry("1280x780")
    open_windows["attendance"] = win

    def on_close_att():
        open_windows.pop("attendance", None)
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close_att)

    theme = THEMES[current_theme]
    win.configure(bg=theme["bg"])

    nb = ttk.Notebook(win)
    nb.pack(fill="both", expand=True)

    # Shared camera and frame buffer for all tabs
    cam_shared = cv2.VideoCapture(0)
    cam_shared.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam_shared.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    latest_frame    = [None]
    grabber_running = [True]

    def frame_grabber():
        if not grabber_running[0]:
            return
        ret, frame = cam_shared.read()
        latest_frame[0] = frame.copy() if ret else None
        win.after(20, frame_grabber)

    frame_grabber()


    # ═══════════════════════════════════════
    # --- GROUP TAB ---
    # ═══════════════════════════════════════
    group_tab = tk.Frame(nb, bg=theme["bg"])
    nb.add(group_tab, text="Group")

    top_bar = tk.Frame(group_tab, bg=theme["bg"])
    top_bar.pack(fill="x", padx=16, pady=10)

    tk.Label(top_bar, text="Group Attendance (Multi-Face)",
             font=("Segoe UI", 16, "bold"), fg="#00ffcc", bg=theme["bg"]).pack(side="left")

    btn_reload = tk.Button(top_bar, text="Reload Faces", bg="#2d89ef", fg="white", bd=0)
    btn_reload.pack(side="right", padx=6)

    body = tk.Frame(group_tab, bg=theme["bg"])
    body.pack(fill="both", expand=True, padx=16, pady=10)

    # Left: camera
    cam_frame = tk.Frame(body, bg=theme["bg"])
    cam_frame.pack(side="left", fill="both", expand=True)

    tk.Label(cam_frame, text="Live Camera", font=("Segoe UI", 14, "bold"),
             fg="#9effa0", bg=theme["bg"]).pack(anchor="w")
    lbl_cam = tk.Label(cam_frame, bg=theme["bg"], width=CAM_WIDTH, height=CAM_HEIGHT)
    lbl_cam.pack(pady=10, fill="both", expand=False)
    lbl_cam_status = tk.Label(cam_frame, text="Loading faces…",
                               fg="#b0b0b0", bg=theme["bg"], font=("Segoe UI", 11))
    lbl_cam_status.pack(anchor="w")

    # Right: present list
    side = tk.Frame(body, width=360, bg=theme["bg"])
    side.pack(side="right", fill="y")
    tk.Label(side, text="Marked Present (Today)", font=("Segoe UI", 13, "bold"),
             fg=theme["fg"], bg=theme["bg"]).pack(anchor="w", padx=12, pady=(10, 6))

    cols = ("Reg No", "Name", "Time")
    tv = ttk.Treeview(side, columns=cols, show="headings", height=16)
    for c in cols:
        tv.heading(c, text=c)
        tv.column(c, anchor=tk.CENTER, width=108)
    vs = ttk.Scrollbar(side, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=vs.set)
    tv.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 12))
    vs.pack(side="left", fill="y", padx=(0, 12), pady=(0, 12))

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("Treeview", background=theme["bg"], foreground=theme["fg"],
                    fieldbackground=theme["bg"], rowheight=24, borderwidth=0)
    style.configure("Treeview.Heading", background=theme["bg"],
                    foreground=theme["fg"], font=("Segoe UI", 10, "bold"))

    today = datetime.now().strftime("%Y-%m-%d")
    already_marked = set()

    def refresh_present_list():
        nonlocal already_marked
        tv.delete(*tv.get_children())
        with get_conn() as conn:
            dfp = pd.read_sql_query(
                """
                SELECT s.reg_no, s.name, a.time
                FROM attendance a
                JOIN students s ON s.id = a.student_id
                WHERE a.date = ?
                ORDER BY a.time DESC
                """,
                conn, params=(today,),
            )
        for _, r in dfp.iterrows():
            tv.insert("", tk.END, values=(r["reg_no"], r["name"], r["time"]))
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT student_id FROM attendance WHERE date = ?", (today,))
            already_marked = {int(x[0]) for x in cur.fetchall()}

    refresh_present_list()

    # Group recognition variables
    known_encodings, known_ids, known_labels = [], [], []

    def reload_faces():
        nonlocal known_encodings, known_ids, known_labels
        known_encodings, known_ids, known_labels = load_all_face_encodings()
        lbl_cam_status.config(text=f"Loaded faces: {len(known_ids)} students ready")

    btn_reload.configure(command=reload_faces)
    reload_faces()

    frame_counts   = {}
    process_counter = 0

    def loop_frame_group():
        nonlocal process_counter
        frame = latest_frame[0]
        if frame is None:
            lbl_cam_status.config(text="Camera not available.")
            return lbl_cam.after(150, loop_frame_group)

        display = frame.copy()

        process_counter = (process_counter + 1) % PROCESS_EVERY_N
        if process_counter == 0 and len(known_encodings) > 0:
            # Run InsightFace on a half-size frame for speed
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            faces = FA.get(small)

            for face in faces:
                # Scale bbox back to full frame
                x1, y1, x2, y2 = (int(v * 2) for v in face.bbox)
                left, top, right, bottom = x1, y1, x2, y2
                face_enc = face.embedding

                sims     = face_similarity(known_encodings, face_enc)
                if sims.size == 0:
                    continue
                best_idx = int(sims.argmax())
                best_sim = float(sims[best_idx])
                sid      = int(known_ids[best_idx])
                label    = known_labels[best_idx]
                pct      = best_sim * 100.0
                is_match = best_sim >= SIMILARITY_THRESHOLD

                color = (0, 255, 0) if is_match else (0, 0, 255)
                cv2.rectangle(display, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    display,
                    f"{label} | {pct:.1f}%",
                    (left, max(20, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

                if is_match:
                    frame_counts[sid] = frame_counts.get(sid, 0) + 1
                    if frame_counts[sid] >= GROUP_REQ_CONSEC and sid not in already_marked:
                        now = datetime.now()
                        with get_conn() as conn2:
                            cur2 = conn2.cursor()
                            cur2.execute(
                                "INSERT OR IGNORE INTO attendance (student_id, date, time, match_percentage) "
                                "VALUES (?, ?, ?, ?)",
                                (sid, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), float(pct)),
                            )
                            conn2.commit()
                        already_marked.add(sid)
                        refresh_present_list()
                        lbl_cam_status.config(text=f"Saved: {label}")
                else:
                    frame_counts[sid] = max(0, frame_counts.get(sid, 0) - 1)

        # HUD border
        cv2.rectangle(display, (10, 10), (CAM_WIDTH - 10, CAM_HEIGHT - 10), (0, 200, 200), 2)
        cv2.putText(
            display,
            f"Loaded: {len(known_ids)} | Marked today: {len(already_marked)}",
            (16, CAM_HEIGHT - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2,
        )

        rgb_disp = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_disp).resize((CAM_WIDTH, CAM_HEIGHT))
        imgtk = ImageTk.PhotoImage(img)
        lbl_cam.imgtk = imgtk
        lbl_cam.configure(image=imgtk)
        lbl_cam.after(15, loop_frame_group)

    loop_frame_group()


    # ═══════════════════════════════════════
    # --- COURSE TAB ---
    # ═══════════════════════════════════════
    course_tab = tk.Frame(nb, bg=theme["bg"])
    nb.add(course_tab, text="Course")

    top_bar_c = tk.Frame(course_tab, bg=theme["bg"])
    top_bar_c.pack(fill="x", padx=16, pady=10)

    tk.Label(top_bar_c, text="Course Attendance", font=("Segoe UI", 16, "bold"),
             fg="#00ffcc", bg=theme["bg"]).pack(side="left")

    course_select_var = tk.StringVar()
    course_combo = ttk.Combobox(top_bar_c, textvariable=course_select_var, width=28, state="readonly")
    course_combo.pack(side="left", padx=10)

    def load_courses():
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT course FROM students WHERE course IS NOT NULL "
                        "AND TRIM(course) <> '' ORDER BY course")
            rows = [r[0] for r in cur.fetchall()]
        course_combo["values"] = rows
        if rows:
            course_combo.current(0)

    load_courses()

    btn_load_course_faces = tk.Button(top_bar_c, text="Load Course Faces",
                                      bg="#2d89ef", fg="white", bd=0)
    btn_load_course_faces.pack(side="left", padx=6)

    body_c = tk.Frame(course_tab, bg=theme["bg"])
    body_c.pack(fill="both", expand=True, padx=16, pady=10)

    cam_frame_c = tk.Frame(body_c, bg=theme["bg"])
    cam_frame_c.pack(side="left", fill="both", expand=True)
    tk.Label(cam_frame_c, text="Live Camera", font=("Segoe UI", 14, "bold"),
             fg="#9effa0", bg=theme["bg"]).pack(anchor="w")
    lbl_cam_c = tk.Label(cam_frame_c, bg=theme["bg"], width=CAM_WIDTH, height=CAM_HEIGHT)
    lbl_cam_c.pack(pady=10, fill="both", expand=False)
    lbl_cam_status_c = tk.Label(cam_frame_c,
                                 text="Select a course and click 'Load Course Faces'",
                                 fg="#b0b0b0", bg=theme["bg"], font=("Segoe UI", 11))
    lbl_cam_status_c.pack(anchor="w")

    side_c = tk.Frame(body_c, width=360, bg=theme["bg"])
    side_c.pack(side="right", fill="y")

    title_c = tk.Label(side_c, text="Marked Present (Today, by Course)",
                       font=("Segoe UI", 13, "bold"), fg=theme["fg"], bg=theme["bg"])
    title_c.pack(anchor="w", padx=12, pady=(10, 6))

    cols_c = ("Reg No", "Name", "Time")
    tv_c = ttk.Treeview(side_c, columns=cols_c, show="headings", height=16)
    for c in cols_c:
        tv_c.heading(c, text=c)
        tv_c.column(c, anchor=tk.CENTER, width=108)
    vs_c = ttk.Scrollbar(side_c, orient="vertical", command=tv_c.yview)
    tv_c.configure(yscrollcommand=vs_c.set)
    tv_c.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 12))
    vs_c.pack(side="left", fill="y", padx=(0, 12), pady=(0, 12))

    style2 = ttk.Style()
    try:
        style2.theme_use("clam")
    except Exception:
        pass
    style2.configure("Treeview", background=theme["bg"], foreground=theme["fg"],
                     fieldbackground=theme["bg"], rowheight=24, borderwidth=0)
    style2.configure("Treeview.Heading", background=theme["bg"],
                     foreground=theme["fg"], font=("Segoe UI", 10, "bold"))

    known_enc_c, known_ids_c, known_labels_c = [], [], []
    course_marked    = set()
    frame_counts_c   = {}
    process_counter_c = 0
    today_c = datetime.now().strftime("%Y-%m-%d")

    def refresh_present_list_course(selected_course: str):
        tv_c.delete(*tv_c.get_children())
        course_marked.clear()
        with get_conn() as conn:
            dfp = pd.read_sql_query(
                """
                SELECT s.id AS sid, s.reg_no, s.name, a.time
                FROM attendance a
                JOIN students s ON s.id = a.student_id
                WHERE a.date = ? AND s.course = ?
                ORDER BY a.time DESC
                """,
                conn, params=(today_c, selected_course),
            )
        for _, r in dfp.iterrows():
            tv_c.insert("", tk.END, values=(r["reg_no"], r["name"], r["time"]))
            course_marked.add(int(r["sid"]))

    def load_course_faces():
        nonlocal known_enc_c, known_ids_c, known_labels_c
        course = course_select_var.get().strip()
        if not course:
            messagebox.showwarning("Course", "Please select a course.")
            return
        known_enc_c, known_ids_c, known_labels_c = load_course_face_encodings(course)
        title_c.config(text=f"Marked Present (Today) — {course}")
        lbl_cam_status_c.config(text=f"Loaded {len(known_ids_c)} faces for: {course}")
        frame_counts_c.clear()
        refresh_present_list_course(course)

    btn_load_course_faces.configure(command=load_course_faces)

    def on_course_change(event=None):
        course = course_select_var.get().strip()
        title_c.config(text=f"Marked Present (Today) — {course if course else ''}")
        tv_c.delete(*tv_c.get_children())
        lbl_cam_status_c.config(text="Course changed. Click 'Load Course Faces' to start.")
        known_enc_c.clear(); known_ids_c.clear(); known_labels_c.clear()
        frame_counts_c.clear(); course_marked.clear()

    course_combo.bind("<<ComboboxSelected>>", on_course_change)

    def loop_frame_course():
        nonlocal process_counter_c
        frame = latest_frame[0]
        if frame is None:
            return lbl_cam_c.after(150, loop_frame_course)

        display = frame.copy()

        process_counter_c = (process_counter_c + 1) % PROCESS_EVERY_N
        if process_counter_c == 0 and len(known_enc_c) > 0:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            faces = FA.get(small)

            for face in faces:
                x1, y1, x2, y2 = (int(v * 2) for v in face.bbox)
                left, top, right, bottom = x1, y1, x2, y2
                face_enc = face.embedding

                sims     = face_similarity(known_enc_c, face_enc)
                if sims.size == 0:
                    continue
                best_idx = int(sims.argmax())
                best_sim = float(sims[best_idx])
                sid      = int(known_ids_c[best_idx])
                label    = known_labels_c[best_idx]
                pct      = best_sim * 100.0
                is_match = best_sim >= SIMILARITY_THRESHOLD

                color = (0, 255, 0) if is_match else (0, 0, 255)
                cv2.rectangle(display, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    display,
                    f"{label} | {pct:.1f}%",
                    (left, max(20, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

                if is_match:
                    frame_counts_c[sid] = frame_counts_c.get(sid, 0) + 1
                    if frame_counts_c[sid] >= GROUP_REQ_CONSEC and sid not in course_marked:
                        now = datetime.now()
                        with get_conn() as conn2:
                            cur2 = conn2.cursor()
                            cur2.execute(
                                "INSERT OR IGNORE INTO attendance (student_id, date, time, match_percentage) "
                                "VALUES (?, ?, ?, ?)",
                                (sid, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), float(pct)),
                            )
                            conn2.commit()
                        course_marked.add(sid)
                        try:
                            reg, nm = [x.strip() for x in label.split("|", 1)]
                        except Exception:
                            reg, nm = label, ""
                        tv_c.insert("", tk.END, values=(reg, nm, now.strftime("%H:%M:%S")))
                        lbl_cam_status_c.config(text=f"Saved: {label}")
                else:
                    frame_counts_c[sid] = max(0, frame_counts_c.get(sid, 0) - 1)

        cv2.rectangle(display, (10, 10), (CAM_WIDTH - 10, CAM_HEIGHT - 10), (0, 200, 200), 2)
        cv2.putText(
            display,
            f"Course faces loaded: {len(known_ids_c)} | Marked (today, course): {len(course_marked)}",
            (16, CAM_HEIGHT - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2,
        )

        rgb_disp = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_disp).resize((CAM_WIDTH, CAM_HEIGHT))
        imgtk = ImageTk.PhotoImage(img)
        lbl_cam_c.imgtk = imgtk
        lbl_cam_c.configure(image=imgtk)
        lbl_cam_c.after(15, loop_frame_course)

    loop_frame_course()


    # ═══════════════════════════════════════
    # --- SINGLE TAB ---
    # ═══════════════════════════════════════
    single_tab = tk.Frame(nb, bg=theme["bg"])
    nb.add(single_tab, text="Single")

    left = tk.Frame(single_tab, bg=theme["bg"])
    left.pack(side="left", fill="y", padx=20, pady=20)

    tk.Label(left, text="Single Student", font=("Segoe UI", 18, "bold"),
             fg="#00ffff", bg=theme["bg"]).pack(pady=(0, 10))

    reg_var    = tk.StringVar()
    sid_var    = tk.StringVar()
    name_var   = tk.StringVar()
    course_var = tk.StringVar()
    photo_var  = tk.StringVar()

    tk.Label(left, text="Reg No:", bg=theme["bg"], fg="#00ffff",
             font=("Segoe UI", 11)).pack(anchor="w")
    reg_entry = neon_entry(left, reg_var)
    reg_entry.focus_set()

    btn_row = tk.Frame(left, bg=theme["bg"])
    btn_row.pack(fill="x", pady=(6, 10))

    right_single = tk.Frame(single_tab, bg=theme["bg"])
    right_single.pack(side="right", fill="both", expand=True, padx=20, pady=20)

    tk.Label(right_single, text="Live Camera", font=("Segoe UI", 14, "bold"),
             fg="#00ff88", bg=theme["bg"]).pack(anchor="w")
    preview_label_s = tk.Label(right_single, bg=theme["bg"], width=CAM_WIDTH, height=CAM_HEIGHT)
    preview_label_s.pack(pady=10)
    status_lbl_s = tk.Label(right_single, text="Waiting for student…",
                             fg=theme["fg"], bg=theme["bg"], font=("Segoe UI", 11))
    status_lbl_s.pack(anchor="w")

    stored_encoding = [None]
    running_match   = [False]
    consecutive     = [0]

    def fetch_student_single(regno):
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, name, course, photo_path FROM students WHERE reg_no=?", (regno,))
            rec = cur.fetchone()
        if not rec:
            status_lbl_s.config(text="Student not found.")
            name_var.set(""); course_var.set(""); photo_var.set(""); sid_var.set("")
            return
        sid_var.set(str(rec[0]))
        name_var.set(rec[1] or "")
        course_var.set(rec[2] or "")
        photo_var.set(rec[3] or "")

        enc = ENCODING_CACHE.get(int(rec[0]))
        if enc is None:
            enc = safe_face_encoding_from_file(photo_var.get())
            if enc is not None:
                ENCODING_CACHE[int(rec[0])] = enc
        if enc is None:
            status_lbl_s.config(text="No face found in stored photo.")
            return
        stored_encoding[0] = enc
        status_lbl_s.config(text="Student loaded. Starting live recognition…")
        start_recognition_single()

    def start_qr_scan_single():
        status_lbl_s.config(text="Scanning QR… (hold code in front of camera)")
        def scan_loop():
            frame = latest_frame[0]
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                codes = decode(rgb)
                if codes:
                    data = codes[0].data.decode("utf-8").strip()
                    reg_var.set(data)
                    status_lbl_s.config(text=f"QR: {data}")
                    fetch_student_single(data)
                    return
            preview_label_s.after(30, scan_loop)
        scan_loop()

    def start_recognition_single():
        if stored_encoding[0] is None or not sid_var.get():
            return
        running_match[0] = True
        consecutive[0]   = 0

    def mark_attendance_single(sid, pct):
        today_s   = datetime.now().strftime("%Y-%m-%d")
        now_time = datetime.now().strftime("%H:%M:%S")
        with get_conn() as conn2:
            cur2 = conn2.cursor()
            cur2.execute("SELECT 1 FROM attendance WHERE student_id=? AND date=?", (sid, today_s))
            if cur2.fetchone():
                status_lbl_s.config(text=f"Already marked today: {name_var.get()}")
                return False
            cur2.execute(
                "INSERT INTO attendance (student_id, date, time, match_percentage) VALUES (?, ?, ?, ?)",
                (sid, today_s, now_time, float(pct)),
            )
            conn2.commit()
        status_lbl_s.config(text=f"Attendance saved: {name_var.get()} ({pct:.2f}%)")
        return True

    def loop_frame_single():
        frame = latest_frame[0]
        if frame is None:
            status_lbl_s.config(text="Camera not available.")
            return preview_label_s.after(120, loop_frame_single)

        display = frame.copy()
        msg   = "Ready"
        color = (0, 255, 255)

        if stored_encoding[0] is not None and running_match[0]:
            faces = FA.get(frame)          # InsightFace expects BGR
            if faces:
                face_enc = faces[0].embedding
                sim      = float(face_similarity([stored_encoding[0]], face_enc)[0])
                pct      = sim * 100.0
                is_match = sim >= SIMILARITY_THRESHOLD
                msg   = f"Match: {pct:.2f}%"
                color = (0, 255, 0) if is_match else (0, 0, 255)
                cv2.putText(display, msg, (16, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(display, "MATCH" if is_match else "NO MATCH", (16, 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                if is_match:
                    consecutive[0] += 1
                else:
                    consecutive[0] = max(0, consecutive[0] - 1)

                if consecutive[0] >= REQ_CONSEC:
                    saved = mark_attendance_single(int(sid_var.get()), pct)
                    if saved:
                        running_match[0]   = False
                        stored_encoding[0] = None
            else:
                msg   = "Looking for face…"
                color = (255, 255, 0)

        cv2.rectangle(display, (10, 10), (CAM_WIDTH - 10, CAM_HEIGHT - 10), color, 2)
        rgb_disp = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_disp).resize((CAM_WIDTH, CAM_HEIGHT))
        imgtk = ImageTk.PhotoImage(img)
        preview_label_s.imgtk = imgtk
        preview_label_s.configure(image=imgtk)
        preview_label_s.after(20, loop_frame_single)

    def do_fetch():
        rn = reg_var.get().strip()
        if rn:
            fetch_student_single(rn)
        else:
            messagebox.showerror("Error", "Enter Reg No or use Scan QR")

    neon_button(btn_row, "Fetch",   do_fetch,              bg="#14818f")
    neon_button(btn_row, "Scan QR", start_qr_scan_single,  bg="#884EA0")

    for label, var in [("Name", name_var), ("Course", course_var), ("Photo Path", photo_var)]:
        tk.Label(left, text=f"{label}:", bg=theme["bg"], fg=theme["fg"]).pack(anchor="w", pady=(8, 0))
        neon_entry(left, var, readonly=True, bg=theme["bg"])

    reg_entry.bind("<Return>", lambda e: do_fetch())
    loop_frame_single()

    # Shared close handler (releases camera & stops grabber)
    def on_close():
        grabber_running[0] = False
        try:
            cam_shared.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        open_windows.pop("attendance", None)
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)


# ---------- Reports ----------
def open_reports():
    if "reports" in open_windows:
        win = open_windows["reports"]
        if win.winfo_exists():
            win.deiconify(); win.lift(); win.focus_force()
            return
        open_windows.pop("reports", None)

    win = tk.Toplevel(root)
    win.title("Reports")
    win.geometry("1060x680")
    open_windows["reports"] = win

    def on_close():
        open_windows.pop("reports", None)
        win.destroy()

    theme = THEMES[current_theme]
    win.configure(bg=theme["bg"])
    win.protocol("WM_DELETE_WINDOW", on_close)

    date_var          = tk.StringVar()
    search_var        = tk.StringVar()
    course_filter_var = tk.StringVar()

    def load_dataframe(date_filter=None, search_filter=None, course_filter=None):
        if not os.path.exists(DB_FILE):
            return pd.DataFrame()
        with get_conn() as conn:
            base = (
                "SELECT s.reg_no AS \"Reg No\", s.name AS \"Name\", s.course AS \"Course\","
                " a.date AS \"Date\", a.time AS \"Time\", a.match_percentage AS \"Match %\""
                " FROM attendance a JOIN students s ON a.student_id = s.id"
            )
            conds, params = [], []
            if date_filter:
                conds.append("a.date = ?"); params.append(date_filter)
            if search_filter:
                conds.append("(s.reg_no LIKE ? OR s.name LIKE ?)")
                q = f"%{search_filter}%"; params.extend([q, q])
            if course_filter:
                conds.append("s.course = ?"); params.append(course_filter)
            if conds:
                base += " WHERE " + " AND ".join(conds)
            base += " ORDER BY a.date DESC, a.time DESC"
            return pd.read_sql_query(base, conn, params=params)

    def update_table():
        for r in tree.get_children():
            tree.delete(r)
        df = load_dataframe(
            date_var.get().strip() or None,
            search_var.get().strip() or None,
            course_filter_var.get().strip() or None,
        )
        for _, row in df.iterrows():
            tree.insert("", tk.END, values=(
                row["Reg No"], row["Name"], row["Course"],
                row["Date"], row["Time"], f"{float(row['Match %']):.2f}%",
            ))

    def export_data():
        df = load_dataframe(
            date_var.get().strip() or None,
            search_var.get().strip() or None,
            course_filter_var.get().strip() or None,
        )
        if df.empty:
            return messagebox.showwarning("No Data", "Nothing to export.")
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")],
        )
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                df.to_csv(path, index=False)
            else:
                df.to_excel(path, index=False)
            messagebox.showinfo("Exported", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def refresh_dates_dropdown():
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT date FROM attendance ORDER BY date DESC LIMIT 300")
            rows = [r[0] for r in cur.fetchall()]
        date_combo["values"] = [""] + rows

    def refresh_courses_dropdown():
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT course FROM students WHERE course IS NOT NULL "
                        "AND TRIM(course) <> '' ORDER BY course")
            rows = [""] + [r[0] for r in cur.fetchall()]
        course_combo["values"] = rows

    top_r = tk.Frame(win, bg=theme["bg"])
    top_r.pack(fill="x", padx=12, pady=10)
    tk.Label(top_r, text="Date:", fg="white", bg="#1e1e1e").pack(side="left")
    date_combo = ttk.Combobox(top_r, textvariable=date_var, width=16)
    date_combo.pack(side="left", padx=6)

    tk.Label(top_r, text="Search:", fg="white", bg="#1e1e1e").pack(side="left", padx=(8, 4))
    tk.Entry(top_r, textvariable=search_var, width=26).pack(side="left")

    tk.Label(top_r, text="Course:", fg="white", bg="#1e1e1e").pack(side="left", padx=(8, 4))
    course_combo = ttk.Combobox(top_r, textvariable=course_filter_var, width=20, state="readonly")
    course_combo.pack(side="left")

    tk.Button(top_r, text="Load",            command=update_table,             bg="#4CAF50", fg="white").pack(side="left", padx=8)
    tk.Button(top_r, text="Refresh Dates",   command=refresh_dates_dropdown,   bg="#777",    fg="white").pack(side="left", padx=6)
    tk.Button(top_r, text="Refresh Courses", command=refresh_courses_dropdown, bg="#777",    fg="white").pack(side="left", padx=6)
    tk.Button(top_r, text="Export",          command=export_data,              bg="#2196F3", fg="white").pack(side="left", padx=6)

    cols = ("Reg No", "Name", "Course", "Date", "Time", "Match %")
    style_r = ttk.Style()
    try:
        style_r.theme_use("clam")
    except Exception:
        pass
    style_r.configure("Treeview", background=theme["bg"], foreground=theme["fg"],
                      fieldbackground="#2b2b2b", rowheight=26, borderwidth=0)
    style_r.configure("Treeview.Heading", background="#1f1f1f", foreground="white",
                      font=("Segoe UI", 11, "bold"))
    style_r.map("Treeview", background=[("selected", "#4CAF50")])

    table_frame = tk.Frame(win, bg="#1e1e1e")
    table_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    tree = ttk.Treeview(table_frame, columns=cols, show="headings")
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, anchor=tk.CENTER, width=140)
    vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    tree.pack(fill="both", expand=True)

    refresh_dates_dropdown()
    refresh_courses_dropdown()
    update_table()


# ---------- Tools (view students) ----------
def open_view_students():
    if "tools" in open_windows:
        win = open_windows["tools"]
        if win.winfo_exists():
            win.deiconify(); win.lift(); win.focus_force()
            return
        open_windows.pop("tools", None)

    win = tk.Toplevel(root)
    win.title("Tools")
    win.geometry("880x620")
    open_windows["tools"] = win

    def on_close():
        open_windows.pop("tools", None)
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)
    theme = THEMES[current_theme]
    win.configure(bg=theme["bg"])

    search_frame = tk.Frame(win)
    search_frame.pack(fill="x", padx=10, pady=5)
    tk.Label(search_frame, text="Search (Reg No / Name):").pack(side="left", padx=5)
    search_var = tk.StringVar()
    entry = tk.Entry(search_frame, textvariable=search_var, width=30)
    entry.pack(side="left", padx=5)

    def search_student():
        q = search_var.get().strip()
        for i in tv.get_children():
            tv.delete(i)
        with get_conn() as conn:
            cur = conn.cursor()
            if q:
                cur.execute("SELECT id, reg_no, name, course, mobile FROM students "
                            "WHERE reg_no=? OR name LIKE ?", (q, f"%{q}%"))
            else:
                cur.execute("SELECT id, reg_no, name, course, mobile FROM students ORDER BY reg_no")
            for row in cur.fetchall():
                tv.insert("", tk.END, values=row)

    tk.Button(search_frame, text="Search",   command=search_student,       bg="#2d89ef", fg="white").pack(side="left", padx=5)
    tk.Button(search_frame, text="Show All", command=lambda: load_students(), bg="#5cb85c", fg="white").pack(side="left", padx=5)

    cols = ("ID", "Reg No", "Name", "Course", "Mobile")
    tv = ttk.Treeview(win, columns=cols, show="headings")
    for c in cols:
        tv.heading(c, text=c)
        tv.column(c, width=150)
    tv.pack(fill="both", expand=True, padx=10, pady=10)

    def load_students():
        for i in tv.get_children():
            tv.delete(i)
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, reg_no, name, course, mobile FROM students ORDER BY reg_no")
            for row in cur.fetchall():
                tv.insert("", tk.END, values=row)

    load_students()

    def show_selected():
        sel = tv.selection()
        if not sel:
            return messagebox.showwarning("Warning", "No student selected")
        data = tv.item(sel[0])["values"]
        messagebox.showinfo("Student Details",
            f"ID: {data[0]}\nReg No: {data[1]}\nName: {data[2]}\nCourse: {data[3]}\nMobile: {data[4]}")

    def delete_selected():
        sel = tv.selection()
        if not sel:
            return messagebox.showwarning("Warning", "No student selected")
        data = tv.item(sel[0])["values"]
        sid, reg_no, name = data[0], data[1], data[2]
        if not messagebox.askyesno("Confirm", f"Delete {name} ({reg_no}) and all attendance?"):
            return
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
            cur.execute("DELETE FROM students WHERE id=?", (sid,))
            conn.commit()
        ENCODING_CACHE.pop(int(sid), None)
        tv.delete(sel[0])
        messagebox.showinfo("Done", f"Student {name} removed successfully.")

    def update_selected():
        sel = tv.selection()
        if not sel:
            return messagebox.showwarning("Warning", "No student selected")
        data = tv.item(sel[0])["values"]
        sid, reg_no, name, course, mobile = data

        upd = tk.Toplevel(win)
        upd.title("Update Student")
        upd.geometry("400x300")

        for lbl_text, default in [("Reg No", reg_no), ("Name", name),
                                   ("Course", course), ("Mobile", mobile)]:
            tk.Label(upd, text=f"{lbl_text}:").pack(pady=5)
            e = tk.Entry(upd)
            e.insert(0, default)
            e.pack(pady=5)
            if lbl_text == "Reg No":   reg_e    = e
            elif lbl_text == "Name":   name_e   = e
            elif lbl_text == "Course": course_e = e
            else:                      mobile_e = e

        def save_update():
            new_reg    = reg_e.get().strip()
            new_name   = name_e.get().strip()
            new_course = course_e.get().strip()
            new_mobile = mobile_e.get().strip()
            if not new_reg or not new_name:
                return messagebox.showwarning("Error", "Reg No and Name are required")
            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute("UPDATE students SET reg_no=?, name=?, course=?, mobile=? WHERE id=?",
                            (new_reg, new_name, new_course, new_mobile, sid))
                conn.commit()
            upd.destroy()
            load_students()
            messagebox.showinfo("Updated", f"Student {new_name} updated successfully.")

        tk.Button(upd, text="Save", command=save_update, bg="#5cb85c", fg="white").pack(pady=10)

    tk.Button(win, text="View Selected Details",   command=show_selected,   bg="#f0ad4e", fg="white").pack(pady=5)
    tk.Button(win, text="Delete Selected Student", command=delete_selected, bg="#d9534f", fg="white").pack(pady=5)
    tk.Button(win, text="Update Selected Student", command=update_selected, bg="#0275d8", fg="white").pack(pady=5)


def open_tools_window():
    win = tk.Toplevel(root)
    win.title("Tools")
    win.geometry("880x620")
    theme = THEMES[current_theme]
    win.configure(bg=theme["bg"])

    frm_qr = tk.LabelFrame(win, text="QR Code Generator",
                            bg=theme["bg"], fg=theme["fg"], padx=10, pady=10)
    frm_qr.pack(fill="x", padx=12, pady=(12, 6))
    tk.Label(frm_qr, text="Reg No:", bg=theme["bg"], fg=theme["fg"]).grid(row=0, column=0, sticky="w")
    qr_reg = tk.StringVar()
    tk.Entry(frm_qr, textvariable=qr_reg, width=24,
             bg=theme["bg"], fg=theme["fg"], relief="flat").grid(row=0, column=1, padx=8)

    def gen_qr():
        r = qr_reg.get().strip()
        if not r:
            return
        os.makedirs("qrcodes", exist_ok=True)
        path = os.path.join("qrcodes", f"{r}.png")
        qrcode.make(r).save(path)
        messagebox.showinfo("QR", f"Saved: {path}")

    tk.Button(frm_qr, text="Generate", command=gen_qr,
              bg="#4CAF50", fg="white").grid(row=0, column=2, padx=8)

    frm_db = tk.LabelFrame(win, text="Database Backup / Restore",
                            fg=theme["fg"], bg=theme["bg"], padx=10, pady=10)
    frm_db.pack(fill="x", padx=12, pady=6)

    def backup_db():
        if not os.path.exists(DB_FILE):
            return messagebox.showerror("Error", "DB not found")
        dest = filedialog.asksaveasfilename(
            defaultextension=".db", filetypes=[("SQLite DB", "*.db")])
        if not dest:
            return
        shutil.copy2(DB_FILE, dest)
        messagebox.showinfo("Backup", f"Saved to: {dest}")

    def restore_db():
        src = filedialog.askopenfilename(
            filetypes=[("SQLite DB", "*.db"), ("All files", "*.*")])
        if not src:
            return
        if not messagebox.askyesno("Confirm", "Overwrite current DB?"):
            return
        shutil.copy2(src, DB_FILE)
        messagebox.showinfo("Restore", "Database restored. Restart app if encodings look outdated.")

    tk.Button(frm_db, text="Backup",  command=backup_db,  bg="#2196F3", fg="white").pack(side="left", padx=8)
    tk.Button(frm_db, text="Restore", command=restore_db, bg="#f39c12", fg="white").pack(side="left", padx=8)

    frm_students = tk.LabelFrame(win, text="Export Student List",
                                  fg=theme["fg"], bg=theme["bg"], padx=10, pady=10)
    frm_students.pack(fill="x", padx=12, pady=6)

    def export_students():
        with get_conn() as conn:
            df = pd.read_sql_query(
                "SELECT reg_no AS \"Reg No\", name AS \"Name\","
                " course AS \"Course\", mobile AS \"Mobile\" FROM students ORDER BY reg_no",
                conn,
            )
        if df.empty:
            return messagebox.showinfo("No Data", "No students.")
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")])
        if not path:
            return
        if path.lower().endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            df.to_excel(path, index=False)
        messagebox.showinfo("Exported", f"Saved: {path}")

    tk.Button(frm_students, text="Export Students", command=export_students,
              bg="#8e44ad", fg="white").pack(side="left", padx=8)

    frm_sum = tk.LabelFrame(win, text="Attendance Summary (Per Day)",
                             fg=theme["fg"], bg=theme["bg"], padx=10, pady=10)
    frm_sum.pack(fill="both", expand=True, padx=12, pady=(6, 12))

    sum_cols = ("Date", "Present Count")
    tree_sum = ttk.Treeview(frm_sum, columns=sum_cols, show="headings", height=9)
    for c in sum_cols:
        tree_sum.heading(c, text=c)
        tree_sum.column(c, anchor=tk.CENTER, width=160)
    tree_sum.pack(side="left", fill="both", expand=True)
    vsb2 = ttk.Scrollbar(frm_sum, orient="vertical", command=tree_sum.yview)
    tree_sum.configure(yscrollcommand=vsb2.set)
    vsb2.pack(side="left", fill="y", padx=6)

    def load_summary():
        with get_conn() as conn:
            df = pd.read_sql_query(
                "SELECT date AS \"Date\", COUNT(*) AS \"Present Count\""
                " FROM attendance GROUP BY date ORDER BY date DESC",
                conn,
            )
        for r in tree_sum.get_children():
            tree_sum.delete(r)
        for _, row in df.iterrows():
            tree_sum.insert("", tk.END, values=(row["Date"], int(row["Present Count"])))

    def export_summary():
        with get_conn() as conn:
            df = pd.read_sql_query(
                "SELECT date AS \"Date\", COUNT(*) AS \"Present Count\""
                " FROM attendance GROUP BY date ORDER BY date DESC",
                conn,
            )
        if df.empty:
            return messagebox.showinfo("No Data", "Nothing to export.")
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")])
        if not path:
            return
        if path.lower().endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            df.to_excel(path, index=False)
        messagebox.showinfo("Exported", f"Saved: {path}")

    btns = tk.Frame(frm_sum, bg=theme["bg"])
    btns.pack(side="left", fill="y", padx=10)
    tk.Button(btns, text="Load Summary",    command=load_summary,       bg="#16a085", fg="white").pack(pady=6, fill="x")
    tk.Button(btns, text="Export Summary",  command=export_summary,     bg="#2980b9", fg="white").pack(pady=6, fill="x")
    tk.Button(btns, text="Manage Students", command=open_view_students, bg="#f39c12", fg="white").pack(pady=6, fill="x")

    load_summary()


# ---------- Dashboard ----------
root = tk.Tk()
root.title(APP_TITLE)
try:
    root.state("zoomed")
except Exception:
    root.attributes("-fullscreen", True)
root.configure(bg="#121212")

top = tk.Frame(root, pady=12)
top.pack(fill="x")
if os.path.exists(LOGO_FILE):
    try:
        l = Image.open(LOGO_FILE).resize((86, 86))
        logo_img = ImageTk.PhotoImage(l)
        tk.Label(top, image=logo_img, bg="#121212").pack(side="left", padx=20)
    except Exception:
        tk.Label(top, text="", bg="#121212").pack(side="left", padx=20)
else:
    tk.Label(top, text="", bg="#121212").pack(side="left", padx=20)

tk.Label(top, text="Central University of Andhra Pradesh",
         font=("Helvetica", 28, "bold"), fg="white", bg="#121212").pack(side="left")
tk.Label(top, text=DEVELOPER_TEXT, fg="#B0B0B0", bg="#121212",
         font=("Segoe UI", 11)).pack(side="right", padx=20)

center = tk.Frame(root, bg="#121212")
center.pack(expand=True)

CARD_BG, HOVER_BG = "#1f1f1f", "#2e7d32"
cards = []


def hex_to_rgba(hex_str, alpha=255):
    h = hex_str.lstrip("#")
    if len(h) == 3:
        r = int(h[0]*2, 16); g = int(h[1]*2, 16); b = int(h[2]*2, 16)
    elif len(h) == 6:
        r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    else:
        return (0, 0, 0, alpha)
    return (r, g, b, alpha)


def make_outer_shadow_image(card_w, card_h, card_hex="#ffffff",
                             radius=18, shadow_hex="#000000",
                             blur_radius=18, spread=0, shadow_opacity=110):
    pad     = blur_radius + spread * 5
    total_w = card_w + pad * 2
    total_h = card_h + pad * 2
    img  = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    shadow_box = [pad - spread, pad - spread,
                  pad - spread + card_w + spread*2, pad - spread + card_h + spread*2]
    draw.rounded_rectangle(shadow_box, radius=radius + spread,
                            fill=hex_to_rgba(shadow_hex, shadow_opacity))
    img  = img.filter(ImageFilter.GaussianBlur(blur_radius))
    draw = ImageDraw.Draw(img)
    card_box = [pad, pad, pad + card_w, pad + card_h]
    draw.rounded_rectangle(card_box, radius=radius, fill=hex_to_rgba(card_hex, 255))
    return img


def animate_zoom(shadow_lbl, normal_img, hover_img, zoom_in=True, steps=6, delay=20):
    img1 = normal_img
    img2 = hover_img
    start, end = (1.0, 1.1) if zoom_in else (1.1, 1.0)
    factor_step = (end - start) / steps

    def step(i=0):
        if i > steps:
            return
        scale   = start + i * factor_step
        w, h    = img1.size
        new_w, new_h = int(w*scale), int(h*scale)
        resized = img2.resize((new_w, new_h), Image.LANCZOS)
        tk_img  = ImageTk.PhotoImage(resized)
        shadow_lbl.configure(image=tk_img)
        shadow_lbl.image = tk_img
        shadow_lbl.after(delay, lambda: step(i+1))

    step()


def make_card(parent, icon_text, label_text, command, card_w=300, card_h=170):
    theme  = THEMES[current_theme]
    radius = 18

    normal_img = make_outer_shadow_image(
        card_w, card_h,
        card_hex=theme["card_bg"],
        radius=radius,
        shadow_hex=theme["shadow"],
        blur_radius=theme.get("shadow_blur", 28),
        spread=0,
        shadow_opacity=theme.get("shadow_opacity", 110),
    )
    normal_tk  = ImageTk.PhotoImage(normal_img)
    shadow_lbl = tk.Label(parent, image=normal_tk, bd=0, bg=theme["bg"])
    shadow_lbl.image = normal_tk
    shadow_lbl.pack_propagate(False)

    inner = tk.Frame(shadow_lbl, bg=theme["card_bg"])
    inner.place(relx=0.5, rely=0.5, anchor="center")

    icon = tk.Label(inner, text=icon_text, font=("Segoe UI Emoji", 36),
                    bg=theme["card_bg"], fg=theme["card_fg"])
    icon.pack(pady=(0, 6))
    lbl  = tk.Label(inner, text=label_text, font=("Segoe UI", 14, "bold"),
                    bg=theme["card_bg"], fg=theme["card_fg"])
    lbl.pack()

    for w in (shadow_lbl, inner, icon, lbl):
        w.bind("<Button-1>", lambda ev: command())

    cards.append({
        "shadow_lbl": shadow_lbl, "normal_img": normal_img,
        "icon": icon, "lbl": lbl, "inner": inner,
        "w": card_w, "h": card_h,
    })
    return shadow_lbl


bottom = tk.Frame(root, pady=10)
bottom.pack(fill="x", side="bottom")


def apply_theme():
    theme = THEMES[current_theme]
    root.configure(bg=theme["bg"])

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview", background=theme["tree_bg"], foreground=theme["tree_fg"],
                    fieldbackground=theme["tree_bg"], rowheight=24, borderwidth=0)
    style.configure("Treeview.Heading", background=theme["tree_header_bg"],
                    foreground=theme["tree_header_fg"], font=("Segoe UI", 10, "bold"))
    style.map("Treeview", background=[("selected", theme["hover_bg"])])
    style.configure("TCombobox", fieldbackground=theme["combo_bg"],
                    background=theme["combo_bg"], foreground=theme["combo_fg"])

    for bar in (top, bottom):
        try:
            bar.configure(bg=theme["bg"])
        except Exception:
            pass
        for widget in bar.winfo_children():
            try:
                widget.configure(bg=theme["bg"], fg=theme["fg"])
            except Exception:
                pass

    for cinfo in cards:
        shadow_lbl = cinfo["shadow_lbl"]
        cw, ch     = cinfo["w"], cinfo["h"]

        normal_img = make_outer_shadow_image(
            cw, ch,
            card_hex=theme["card_bg"],
            radius=18,
            shadow_hex=theme.get("shadow", "#000000"),
            blur_radius=theme.get("shadow_blur", 28),
            spread=10,
            shadow_opacity=theme.get("shadow_opacity", 110),
        )
        normal_tk = ImageTk.PhotoImage(normal_img)
        shadow_lbl.configure(image=normal_tk, bg=theme["bg"])
        shadow_lbl.image  = normal_tk
        cinfo["normal_img"] = normal_img
        cinfo["inner"].configure(bg=theme["card_bg"])
        cinfo["icon"].configure(bg=theme["card_bg"], fg=theme["card_fg"])
        cinfo["lbl"].configure(bg=theme["card_bg"],  fg=theme["card_fg"])


grid = tk.Frame(center)
grid.pack()
make_card(grid, "📝", "Enroll Student", open_enrollment).grid(row=0, column=0, padx=0, pady=0)
make_card(grid, "👥", "Attendance",     open_attendance).grid(row=0, column=1, padx=0, pady=0)
make_card(grid, "📊", "Reports",        open_reports).grid(   row=1, column=0, padx=0, pady=0)
make_card(grid, "🛠", "Tools",          open_tools_window).grid(row=1, column=1, padx=0, pady=0)

apply_theme()


def exit_app():
    if messagebox.askyesno("Exit", "Close the dashboard?"):
        root.destroy()


def toggle_theme():
    global current_theme
    current_theme = "light" if current_theme == "dark" else "dark"
    apply_theme()


tk.Label(bottom, text=DEVELOPER_TEXT, fg="#B0B0B0", bg="#121212",
         font=("Segoe UI", 11)).pack(side="left", padx=20)
tk.Button(bottom, text="Toggle Theme", command=toggle_theme,
          bg="#555", fg="white").pack(side="right", padx=18)
tk.Button(bottom, text="Exit", command=exit_app,
          bg="#d32f2f", fg="white").pack(side="right", padx=18)

root.bind("<Escape>", lambda e: exit_app())
root.bind("e", lambda e: open_enrollment())
root.bind("a", lambda e: open_attendance())
root.bind("r", lambda e: open_reports())
root.bind("t", lambda e: open_tools_window())

root.mainloop()
