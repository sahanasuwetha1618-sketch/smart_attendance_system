import logging
import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import plotly.express as px
import pyttsx3
import json, os, io, threading, queue
from datetime import datetime, date, timedelta
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

logging.getLogger("MediaFileHandler").setLevel(logging.ERROR)
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore[import]
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ADMIN_DEFAULT = {"username": "admin", "password": "admin123", "role": "admin",
                 "full_name": "Administrator", "phone": "", "parent_phone": "",
                 "reg_date": str(date.today())}
DATA_DIR       = "data"
USERS_FILE     = os.path.join(DATA_DIR, "users.json")
ATT_FILE       = os.path.join(DATA_DIR, "attendance.csv")
SMS_FILE       = os.path.join(DATA_DIR, "sms_config.json")
HOLIDAYS_FILE  = os.path.join(DATA_DIR, "holidays.json")
FACE_DIRS      = {"admin": "admin_faces", "staff": "staff_faces", "student": "student_faces"}
UNKNOWN_DIR    = "unknown_faces"
# Face recognition tolerance: lower = stricter. 0.45 is a good balance.
FACE_TOLERANCE = 0.45

for d in list(FACE_DIRS.values()) + [UNKNOWN_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── SMS CONFIG ────────────────────────────────────────────────────────────────
def load_sms_config():
    defaults = {"sid": os.getenv("TWILIO_SID", ""), "token": os.getenv("TWILIO_TOKEN", ""),
                "from_number": os.getenv("TWILIO_FROM", ""),
                "admin_phone": os.getenv("ADMIN_PHONE", ""),
                "enabled": False, "threshold": 75}
    if not os.path.exists(SMS_FILE):
        return defaults
    with open(SMS_FILE) as f:
        cfg = json.load(f)
    defaults.update(cfg)
    return defaults

def save_sms_config(cfg):
    with open(SMS_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# ─── USER HELPERS ──────────────────────────────────────────────────────────────
def load_users():
    if not os.path.exists(USERS_FILE):
        data = {"admin": ADMIN_DEFAULT}
        save_users(data)
        return data
    with open(USERS_FILE) as f:
        users = json.load(f)
    if "admin" not in users:
        users["admin"] = ADMIN_DEFAULT
    # Back-fill reg_date for existing users that don't have it.
    # Use their earliest attendance record date so existing data stays correct.
    changed = False
    if os.path.exists(ATT_FILE):
        try:
            att_df = pd.read_csv(ATT_FILE, dtype=str)
            att_df["date"] = att_df["date"].astype(str).str.strip()
        except Exception:
            att_df = pd.DataFrame()
    else:
        att_df = pd.DataFrame()

    for uname, udata in users.items():
        if "reg_date" not in udata or not udata["reg_date"]:
            # Try to find earliest attendance record for this user
            if not att_df.empty and "username" in att_df.columns:
                user_rows = att_df[att_df["username"] == uname]["date"]
                if not user_rows.empty:
                    earliest = sorted(user_rows.tolist())[0]
                    udata["reg_date"] = earliest
                else:
                    udata["reg_date"] = str(date.today())
            else:
                udata["reg_date"] = str(date.today())
            changed = True

    if changed:
        save_users(users)
    return users

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

# ─── HOLIDAY HELPERS ──────────────────────────────────────────────────────────
def load_holidays() -> set:
    """Return a set of date objects that are marked as holidays."""
    if not os.path.exists(HOLIDAYS_FILE):
        return set()
    with open(HOLIDAYS_FILE) as f:
        raw = json.load(f)
    result = set()
    for d in raw:
        try:
            result.add(date.fromisoformat(d))
        except ValueError:
            pass
    return result

def save_holidays(holidays: set):
    with open(HOLIDAYS_FILE, "w") as f:
        json.dump(sorted(str(d) for d in holidays), f, indent=2)

def is_working_day(d: date, holidays: set) -> bool:
    """A working day is Mon–Fri and not a holiday."""
    return d.weekday() < 5 and d not in holidays

def working_days_between(start: date, end: date, holidays: set) -> list:
    """
    Return list of working day dates from start to end (inclusive).
    Excludes weekends (Sat/Sun) and holidays.
    """
    days = []
    current = start
    while current <= end:
        if is_working_day(current, holidays):
            days.append(current)
        current += timedelta(days=1)
    return days

# ─── ATTENDANCE HELPERS ────────────────────────────────────────────────────────
def load_attendance():
    if not os.path.exists(ATT_FILE):
        return pd.DataFrame(columns=["name", "username", "date", "status"])
    df = pd.read_csv(ATT_FILE, dtype=str)
    df["date"] = df["date"].astype(str).str.strip()
    return df

def save_attendance(df):
    df["date"] = df["date"].astype(str).str.strip()
    df.to_csv(ATT_FILE, index=False)

def mark_attendance(username, full_name):
    """
    Mark Present for today.
    - Skips weekends and holidays.
    - If an Absent record already exists for today (from mark_daily_absents),
      it is updated to Present — ensuring only one record per day.
    - Returns True if status changed to Present, False if already Present.
    """
    holidays  = load_holidays()
    today     = date.today()
    if not is_working_day(today, holidays):
        return False
    today_str = str(today)
    df        = load_attendance()
    mask      = (df["username"] == username) & (df["date"] == today_str)

    if mask.any():
        if (df.loc[mask, "status"] == "Present").all():
            return False  # already present, nothing to do
        # Update Absent → Present
        df.loc[mask, "status"] = "Present"
        save_attendance(df)
        return True

    # No record yet — insert Present
    new_row = pd.DataFrame([{"name": full_name, "username": username,
                              "date": today_str, "status": "Present"}])
    df = pd.concat([df, new_row], ignore_index=True)
    save_attendance(df)
    return True

def mark_daily_absents():
    """
    Idempotent — safe to call multiple times per day.
    For every student whose reg_date <= today and today is a working day,
    insert an Absent record if no record exists yet for today.
    This guarantees exactly one record per student per working day.
    """
    holidays  = load_holidays()
    today     = date.today()
    if not is_working_day(today, holidays):
        return
    today_str = str(today)
    users     = load_users()
    df        = load_attendance()
    changed   = False
    for uname, udata in users.items():
        if udata.get("role") != "student":
            continue
        try:
            reg = date.fromisoformat(udata.get("reg_date", today_str))
        except ValueError:
            reg = today
        if today < reg:
            continue  # not yet registered
        already = not df[(df["username"] == uname) & (df["date"] == today_str)].empty
        if not already:
            new_row = pd.DataFrame([{"name": udata["full_name"], "username": uname,
                                     "date": today_str, "status": "Absent"}])
            df      = pd.concat([df, new_row], ignore_index=True)
            changed = True
    if changed:
        save_attendance(df)

def attendance_stats(username):
    """
    Attendance calculation rules:
    - total   = calendar days from reg_date to today (inclusive), excl. weekends & holidays
    - present = days with a 'Present' record on/after reg_date
    - absent  = total - present
    - pct     = present / total * 100

    Example:
      reg_date=18/03, present on 18/03, absent on 19/03 → total=2, present=1, absent=1, 50%
      reg_date=19/03, no face seen on 19/03             → total=1, present=0, absent=1, 0%
    """
    users        = load_users()
    reg_date_str = users.get(username, {}).get("reg_date", str(date.today()))
    try:
        reg_date = date.fromisoformat(reg_date_str)
    except ValueError:
        reg_date = date.today()

    holidays  = load_holidays()
    today     = date.today()

    # Total = working days from reg_date to today inclusive
    work_days  = working_days_between(reg_date, today, holidays)
    total_days = len(work_days)

    df  = load_attendance()
    udf = df[df["username"] == username].copy()

    if udf.empty:
        return {"total": total_days, "present": 0, "absent": total_days,
                "pct": 0.0, "weekly": 0, "monthly": 0, "reg_date": str(reg_date)}

    udf["date_parsed"] = pd.to_datetime(udf["date"], errors="coerce")

    # Only records on/after registration date
    udf = udf[udf["date_parsed"].dt.date >= reg_date].copy()

    # Deduplicate per day — Present beats Absent if both somehow exist
    udf = udf.sort_values(["date_parsed", "status"], ascending=[True, True])
    udf = udf.drop_duplicates(subset=["date_parsed"], keep="last")

    present = int((udf["status"] == "Present").sum())
    absent  = max(total_days - present, 0)
    pct     = round(present / total_days * 100, 1) if total_days else 0.0

    weekly  = int(udf[udf["date_parsed"].dt.isocalendar().week == today.isocalendar()[1]].shape[0])
    monthly = int(udf[udf["date_parsed"].dt.month == today.month].shape[0])

    return {"total": total_days, "present": present, "absent": absent,
            "pct": pct, "weekly": weekly, "monthly": monthly,
            "reg_date": str(reg_date)}

# ─── MARKS HELPERS ────────────────────────────────────────────────────────────
INTERNAL_FILE   = os.path.join(DATA_DIR, "internal_marks.json")
ASSIGNMENT_FILE = os.path.join(DATA_DIR, "assignment_marks.json")
SEMESTER_FILE   = os.path.join(DATA_DIR, "semester_marks.json")

def load_internal_marks():
    if not os.path.exists(INTERNAL_FILE): return {}
    with open(INTERNAL_FILE) as f: return json.load(f)

def save_internal_marks(data):
    with open(INTERNAL_FILE, "w") as f: json.dump(data, f, indent=2)

def load_assignment_marks():
    if not os.path.exists(ASSIGNMENT_FILE): return {}
    with open(ASSIGNMENT_FILE) as f: return json.load(f)

def save_assignment_marks(data):
    with open(ASSIGNMENT_FILE, "w") as f: json.dump(data, f, indent=2)

def load_semester_marks():
    if not os.path.exists(SEMESTER_FILE): return {}
    with open(SEMESTER_FILE) as f: return json.load(f)

def save_semester_marks(data):
    with open(SEMESTER_FILE, "w") as f: json.dump(data, f, indent=2)

def internal_marks_summary(username):
    data    = load_internal_marks()
    student = data.get(username, {})
    if not student:
        return {"total": 0, "avg": 0, "pct": 0, "subjects": {}}
    subjects, total, max_total = {}, 0, 0
    for subj, marks in student.items():
        i1, i2, i3 = (float(marks.get(k, 0) or 0) for k in ("internal1","internal2","internal3"))
        st_ = i1 + i2 + i3
        subjects[subj] = {"internal1": i1, "internal2": i2, "internal3": i3,
                          "total": st_, "avg": st_/3 if st_ else 0,
                          "pct": round(st_/300*100, 1)}
        total += st_; max_total += 300
    avg = total / (len(subjects)*3) if subjects else 0
    return {"total": total, "avg": avg, "pct": round(total/max_total*100,1) if max_total else 0,
            "subjects": subjects}

def assignment_marks_summary(username):
    data    = load_assignment_marks()
    student = data.get(username, {})
    if not student:
        return {"total": 0, "avg": 0, "pct": 0, "subjects": {}}
    subjects, grand_total, grand_max = {}, 0, 0
    for subj, entries in student.items():
        scores  = [float(v or 0) for v in entries.values()]
        s_total = sum(scores); s_max = len(scores)*100
        subjects[subj] = {"scores": entries, "total": s_total, "max": s_max,
                          "avg": s_total/len(scores) if scores else 0,
                          "pct": round(s_total/s_max*100,1) if s_max else 0}
        grand_total += s_total; grand_max += s_max
    avg = grand_total/grand_max*100 if grand_max else 0
    return {"total": grand_total, "max": grand_max, "avg": round(avg,1),
            "pct": round(grand_total/grand_max*100,1) if grand_max else 0, "subjects": subjects}

def semester_marks_summary(username):
    data    = load_semester_marks()
    student = data.get(username, {})
    if not student:
        return {"subjects": {}}
    subjects = {}
    for subj, grade in student.items():
        subjects[subj] = {"grade": grade, "points": grade_to_points(grade)}
    return {"subjects": subjects}

# Grade scale helpers
GRADE_OPTIONS = ["O", "A+", "A", "B+", "B", "C", "U"]

# Map display label → stored key (same since no ranges shown)
GRADE_LABELS = {g: g for g in GRADE_OPTIONS}

# Grade → midpoint marks (for any numeric summary if needed)
GRADE_POINTS = {"O": 95, "A+": 85, "A": 75, "B+": 65, "B": 55, "C": 47, "U": 0}

def grade_to_points(grade: str) -> int:
    return GRADE_POINTS.get(grade, 0)

def grade_display_label(grade: str) -> str:
    return grade

# ─── REPORT HELPERS ────────────────────────────────────────────────────────────
def to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def to_excel(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Attendance")
    return buf.getvalue()

# ─── SMS / VOICE ───────────────────────────────────────────────────────────────
def _twilio_error_message(exc):
    if isinstance(exc, TwilioRestException):
        if exc.code == 21608:
            return "Twilio trial: verify destination number in Twilio console."
        return f"Twilio error {exc.code}: {exc.msg or exc}"
    return str(exc)

def send_sms_result(to, body):
    """Send SMS via Twilio. Returns (success, error_msg)."""
    cfg = load_sms_config()
    if not cfg.get("enabled"):
        return False, "SMS not enabled"
    sid, token, from_ = cfg.get("sid",""), cfg.get("token",""), cfg.get("from_number","")
    if not (sid and token and from_ and to and to.strip()):
        return False, "Missing credentials or phone number"
    try:
        Client(sid, token).messages.create(body=body, from_=from_, to=to.strip())
        return True, ""
    except Exception as e:
        return False, _twilio_error_message(e)

def send_low_attendance_alerts(uname, fn, stats, users):
    """
    Send low-attendance SMS to: student, parent, all staff, admin.
    Uses the configurable threshold from sms_config.
    """
    cfg       = load_sms_config()
    threshold = int(cfg.get("threshold", 75))
    if stats["pct"] >= threshold or stats["total"] == 0:
        return
    msg_student = (f"Dear {fn}, your attendance is {stats['pct']}% "
                   f"({stats['present']}/{stats['total']} days). "
                   f"Minimum required is {threshold}%.")
    msg_parent  = (f"Parent Alert: {fn}'s attendance is {stats['pct']}% "
                   f"({stats['present']}/{stats['total']} days), below {threshold}%.")
    msg_staff   = (f"Low Attendance Alert: {fn} is at {stats['pct']}% "
                   f"({stats['present']}/{stats['total']} days).")
    # Student
    send_sms_result(users.get(uname, {}).get("phone", ""), msg_student)
    # Parent
    send_sms_result(users.get(uname, {}).get("parent_phone", ""), msg_parent)
    # Admin
    admin_phone = cfg.get("admin_phone", "")
    if admin_phone:
        send_sms_result(admin_phone, msg_staff)
    # All staff
    for sv in users.values():
        if sv.get("role") == "staff" and sv.get("phone"):
            send_sms_result(sv["phone"], msg_staff)

def speak(text):
    def _run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass  # silently skip on cloud (no audio hardware)
    threading.Thread(target=_run, daemon=True).start()

# ─── FACE HELPERS ──────────────────────────────────────────────────────────────
def load_known_faces():
    """
    Load all face encodings from disk.
    For each user, average their multiple encodings into one representative
    encoding — this improves matching accuracy significantly.
    """
    from collections import defaultdict
    enc_map  = defaultdict(list)  # username -> list of encodings
    role_map = {}                 # username -> role

    for role, folder in FACE_DIRS.items():
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            path = os.path.join(folder, fname)
            img  = face_recognition.load_image_file(path)
            # Use CNN model for better accuracy if GPU available, else HOG
            encs = face_recognition.face_encodings(img, num_jitters=2)
            if encs:
                base  = os.path.splitext(fname)[0]
                uname = "_".join(base.split("_")[:-1]) if "_" in base else base
                enc_map[uname].append(encs[0])
                role_map[uname] = role

    encodings, names, roles = [], [], []
    for uname, encs in enc_map.items():
        # Average all encodings for this user → more robust single encoding
        avg_enc = np.mean(encs, axis=0)
        encodings.append(avg_enc)
        names.append(uname)
        roles.append(role_map[uname])

    return encodings, names, roles

def capture_face_images(username, role, num_shots=5):
    """
    Capture face images with quality checks:
    - Only save if face is detected and image brightness is acceptable.
    """
    folder  = FACE_DIRS[role]
    cap     = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    saved   = 0
    stframe = st.empty()
    status  = st.empty()

    while saved < num_shots:
        ret, frame = cap.read()
        if not ret:
            break
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check brightness — skip very dark or overexposed frames
        brightness = gray.mean()
        locs = face_recognition.face_locations(rgb, model="hog")

        for top, right, bottom, left in locs:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        stframe.image(rgb, channels="RGB")

        if locs and 40 < brightness < 220:
            cv2.imwrite(os.path.join(folder, f"{username}_{saved}.jpg"), frame)
            saved += 1
            status.info(f"Captured {saved}/{num_shots}  (brightness: {brightness:.0f})")
        elif locs and not (40 < brightness < 220):
            status.warning(f"Poor lighting (brightness={brightness:.0f}). Adjust and hold still.")

    cap.release()
    stframe.empty()
    status.success(f"Face capture complete for {username} ({saved} images saved).")

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("username",  "")
st.session_state.setdefault("role",      "")
st.session_state.setdefault("full_name", "")

# ─── LOGIN ─────────────────────────────────────────────────────────────────────
def page_login():
    st.title("Smart Attendance System")
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        users = load_users()
        if username in users and users[username]["password"] == password:
            u = users[username]
            st.session_state.logged_in = True
            st.session_state.username  = username
            st.session_state.role      = u["role"]
            st.session_state.full_name = u["full_name"]
            st.rerun()
        else:
            st.error("Invalid credentials.")

# ─── REGISTER ──────────────────────────────────────────────────────────────────
def page_register(reg_role):
    st.subheader(f"Register {reg_role.capitalize()}")
    full_name    = st.text_input("Full Name")
    username     = st.text_input("Username")
    password     = st.text_input("Password", type="password")
    phone        = st.text_input("Phone Number")        if reg_role in ("admin", "student", "staff") else ""
    parent_phone = st.text_input("Parent Phone Number") if reg_role == "student" else ""
    if st.button("Register"):
        if not (full_name and username and password):
            st.error("Fill all required fields.")
            return
        users = load_users()
        if username in users:
            st.error("Username already exists.")
            return
        # Store registration date — used for correct absent-day calculation
        users[username] = {"full_name": full_name, "username": username,
                           "password": password, "role": reg_role,
                           "phone": phone, "parent_phone": parent_phone,
                           "reg_date": str(date.today())}
        save_users(users)
        st.success(f"{reg_role.capitalize()} '{username}' registered on {date.today()}.")

# ─── FACE CAPTURE ──────────────────────────────────────────────────────────────
def page_face_capture():
    st.subheader("Capture Face")
    users     = load_users()
    all_users = [(k, v["full_name"], v["role"]) for k, v in users.items()]
    options   = [f"{fn} ({u}) [{r}]" for u, fn, r in all_users]
    choice    = st.selectbox("Select User", options)
    if st.button("Start Capture"):
        idx             = options.index(choice)
        uname, _, urole = all_users[idx]
        capture_face_images(uname, urole)

# ─── LIVE ATTENDANCE ───────────────────────────────────────────────────────────
def page_live_attendance():
    st.subheader("Live Camera Attendance")
    run     = st.checkbox("Start Camera")
    stframe = st.empty()

    if not run:
        return

    # Mark absent for all students who haven't been seen today (idempotent)
    mark_daily_absents()

    known_encs, known_names, known_roles = load_known_faces()
    users            = load_users()
    notified_unknown = set()

    # Shared state between camera loop and recognition thread
    frame_queue  = queue.Queue(maxsize=1)
    overlay_lock = threading.Lock()
    overlay_data = []  # list of (top, right, bottom, left, label, color)

    def recognition_worker():
        """Background thread: runs face recognition without blocking the camera."""
        while True:
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            if frame is None:
                break

            # Resize to 1/4 for faster detection, then scale coords back
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            # Apply CLAHE to improve recognition under poor lighting
            lab   = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l     = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            rgb_enh  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

            locs = face_recognition.face_locations(rgb_enh, model="hog")
            encs = face_recognition.face_encodings(rgb_enh, locs, num_jitters=1)

            new_overlay = []
            for enc, (top, right, bottom, left) in zip(encs, locs):
                top *= 4; right *= 4; bottom *= 4; left *= 4
                label = "Unknown"; color = (0, 0, 255)

                if known_encs:
                    distances = face_recognition.face_distance(known_encs, enc)
                    best_idx  = int(np.argmin(distances))
                    best_dist = distances[best_idx]

                    # Use distance-based matching with configurable tolerance
                    if best_dist < FACE_TOLERANCE:
                        uname = known_names[best_idx]
                        urole = known_roles[best_idx]
                        fn    = users.get(uname, {}).get("full_name", uname)
                        conf  = round((1 - best_dist) * 100, 1)

                        if urole == "student":
                            color = (0, 255, 0)
                            if mark_attendance(uname, fn):
                                label = f"{fn} ({conf}%) | Marked"
                                speak(f"{fn}, attendance marked.")
                                stats = attendance_stats(uname)
                                # Send low-attendance alerts to all parties
                                send_low_attendance_alerts(uname, fn, stats, users)
                            else:
                                label = f"{fn} ({conf}%) | Already Marked"
                        elif urole == "admin":
                            color = (255, 165, 0)
                            label = f"{fn} ({conf}%) | Admin"
                        else:
                            color = (255, 255, 0)
                            label = f"{fn} ({conf}%) | Staff"
                else:
                    # Unknown face — save snapshot and alert admin
                    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                    key = f"{left}_{top}"
                    if key not in notified_unknown:
                        notified_unknown.add(key)
                        cv2.imwrite(os.path.join(UNKNOWN_DIR, f"unknown_{ts}.jpg"), frame)
                        admin_phone = load_sms_config().get("admin_phone", "")
                        if admin_phone:
                            send_sms_result(admin_phone, f"Unknown person detected at {ts}.")

                new_overlay.append((top, right, bottom, left, label, color))

            with overlay_lock:
                overlay_data.clear()
                overlay_data.extend(new_overlay)

    worker = threading.Thread(target=recognition_worker, daemon=True)
    worker.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    frame_count = 0
    RECOG_EVERY = 5  # run recognition every 5th frame for smooth video

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % RECOG_EVERY == 0 and not frame_queue.full():
            frame_queue.put(frame.copy())

        display = frame.copy()
        with overlay_lock:
            current_overlay = list(overlay_data)
        for (top, right, bottom, left, label, color) in current_overlay:
            cv2.rectangle(display, (left, top), (right, bottom), color, 2)
            cv2.putText(display, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        stframe.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), channels="RGB")

    frame_queue.put(None)
    cap.release()

# ─── ADMIN/STAFF DASHBOARD ─────────────────────────────────────────────────────
def page_dashboard_admin_staff():
    st.subheader("Attendance Dashboard")
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=5000, key="admin_refresh")

    # Ensure today's absent records exist for all students
    mark_daily_absents()

    df    = load_attendance()
    users = load_users()
    students = [u for u, v in users.items() if v["role"] == "student"]

    today_str      = str(date.today())
    total_students = len(students)
    today_df       = df[df["date"] == today_str] if not df.empty else pd.DataFrame()
    today_present  = int((today_df["status"] == "Present").sum()) if not today_df.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students",  total_students)
    c2.metric("Present Today",   today_present)
    c3.metric("Absent Today",    max(total_students - today_present, 0))
    c4.metric("All-time Present", int((df["status"] == "Present").sum()) if not df.empty else 0)

    if not students:
        st.info("No students registered.")
        return

    # Build per-student summary rows
    rows = []
    for uname in students:
        fn       = users[uname]["full_name"]
        stats    = attendance_stats(uname)
        internal = internal_marks_summary(uname)
        assign   = assignment_marks_summary(uname)
        sem      = semester_marks_summary(uname)
        rows.append({
            "Name": fn, "Username": uname,
            "Reg Date": stats["reg_date"],
            "Total Days": stats["total"], "Present": stats["present"],
            "Absent": stats["absent"], "Weekly": stats["weekly"],
            "Monthly": stats["monthly"], "Attendance %": stats["pct"],
            "Internal %": internal["pct"], "Assignment %": assign["pct"],
            "Semester Grades": ", ".join(
                f"{s}:{v['grade']}" for s, v in sem["subjects"].items()
            ) if sem["subjects"] else "—",
        })
    rdf = pd.DataFrame(rows)

    st.divider()
    st.markdown("#### Per-Student Summary")

    def highlight_low(val):
        if isinstance(val, (int, float)) and val < 75:
            return "background-color:#ffcccc;color:#900"
        return ""

    st.dataframe(rdf.style.applymap(highlight_low, subset=["Attendance %"]),
                 width='stretch')

    # ── Charts ──
    fig = px.bar(rdf, x="Name", y="Attendance %", color="Attendance %",
                 color_continuous_scale=["red","orange","green"],
                 range_color=[0,100], title="Attendance % per Student")
    fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="75% threshold")
    st.plotly_chart(fig, width='stretch')

    fig2 = px.bar(rdf, x="Name", y=["Present","Absent"],
                  barmode="group", title="Present vs Absent per Student")
    st.plotly_chart(fig2, width='stretch')

    # ── Edit Attendance ──
    with st.expander("✏️ Edit Per-Student Attendance"):
        edit_students = {v["full_name"]: k for k, v in users.items() if v["role"] == "student"}
        sel_name  = st.selectbox("Student", list(edit_students.keys()), key="dash_edit_sel")
        sel_uname = edit_students[sel_name]
        udf = df[df["username"] == sel_uname].sort_values("date", ascending=False)
        if udf.empty:
            st.info("No records yet.")
        else:
            st.dataframe(udf[["name","date","status"]].reset_index(drop=True), width='stretch')
        sel_date = st.date_input("Date", value=date.today(), key="dash_edit_date")
        ds       = str(sel_date)
        action   = st.radio("Action", ["Add Present","Delete Record"],
                            key="dash_edit_action", horizontal=True)
        rec_exists = not df[(df["username"] == sel_uname) & (df["date"] == ds)].empty
        if action == "Add Present" and rec_exists:
            st.warning(f"Record already exists for {sel_name} on {ds}.")
        if action == "Delete Record" and not rec_exists:
            st.warning(f"No record found for {sel_name} on {ds}.")
        if st.button("Save", key="dash_edit_save"):
            fresh   = load_attendance()
            rec_now = not fresh[(fresh["username"] == sel_uname) & (fresh["date"] == ds)].empty
            if action == "Add Present":
                if rec_now:
                    st.warning("Record already exists.")
                else:
                    fresh = pd.concat([fresh, pd.DataFrame([{"name": sel_name,
                        "username": sel_uname, "date": ds, "status": "Present"}])],
                        ignore_index=True)
                    save_attendance(fresh)
                    st.success(f"Added attendance for {sel_name} on {ds}.")
                    st.rerun()
            else:
                mask = (fresh["username"] == sel_uname) & (fresh["date"] == ds)
                if mask.sum() == 0:
                    st.warning("No record found.")
                else:
                    save_attendance(fresh[~mask].reset_index(drop=True))
                    st.success(f"Deleted record for {sel_name} on {ds}.")
                    st.rerun()

    # ── Marks Management ──
    with st.expander("📚 Manage Marks (Internal / Assignment / Semester)"):
        student_names = [users[u]["full_name"] for u in students]
        sel_student   = st.selectbox("Student", student_names, key="marks_sel_student")
        sel_uname     = [u for u in students if users[u]["full_name"] == sel_student][0]

        mark_tab1, mark_tab2, mark_tab3 = st.tabs(["Internal Marks","Assignment Marks","Semester Marks"])

        with mark_tab1:
            int_data    = load_internal_marks()
            int_student = int_data.get(sel_uname, {})
            subj_int    = st.text_input("Subject", value="Math", key="int_subject")
            ex_int      = int_student.get(subj_int, {})
            i1 = st.number_input("Internal 1 (max 100)", 0.0, 100.0, float(ex_int.get("internal1",0)), key="int_i1")
            i2 = st.number_input("Internal 2 (max 100)", 0.0, 100.0, float(ex_int.get("internal2",0)), key="int_i2")
            i3 = st.number_input("Internal 3 (max 100)", 0.0, 100.0, float(ex_int.get("internal3",0)), key="int_i3")
            ci_save, ci_del = st.columns(2)
            if ci_save.button("Save Internal", key="int_save"):
                int_data.setdefault(sel_uname, {})[subj_int] = {"internal1":i1,"internal2":i2,"internal3":i3}
                save_internal_marks(int_data); st.success("Saved."); st.rerun()
            if ci_del.button("Delete Subject", key="int_del"):
                if sel_uname in int_data and subj_int in int_data[sel_uname]:
                    del int_data[sel_uname][subj_int]; save_internal_marks(int_data)
                    st.success("Deleted."); st.rerun()
                else: st.warning("Subject not found.")
            s_int = internal_marks_summary(sel_uname)
            if s_int["subjects"]:
                st.dataframe(pd.DataFrame([{"Subject":s,"I1":v["internal1"],"I2":v["internal2"],
                    "I3":v["internal3"],"Total":v["total"],"%":v["pct"]}
                    for s,v in s_int["subjects"].items()]), width='stretch')
                st.info(f"Overall — Avg: {round(s_int['avg'],1)} | %: {s_int['pct']}%")

        with mark_tab2:
            asgn_data    = load_assignment_marks()
            asgn_student = asgn_data.get(sel_uname, {})
            subj_asgn    = st.text_input("Subject", value="Math", key="asgn_subject")
            asgn_no      = st.text_input("Assignment No.", value="A1", key="asgn_no")
            ex_asgn      = float(asgn_student.get(subj_asgn, {}).get(asgn_no, 0))
            asgn_score   = st.number_input("Score (max 100)", 0.0, 100.0, ex_asgn, key="asgn_score")
            ca_save, ca_del = st.columns(2)
            if ca_save.button("Save Assignment", key="asgn_save"):
                asgn_data.setdefault(sel_uname,{}).setdefault(subj_asgn,{})[asgn_no] = asgn_score
                save_assignment_marks(asgn_data); st.success("Saved."); st.rerun()
            if ca_del.button("Delete Assignment", key="asgn_del"):
                if (sel_uname in asgn_data and subj_asgn in asgn_data[sel_uname]
                        and asgn_no in asgn_data[sel_uname][subj_asgn]):
                    del asgn_data[sel_uname][subj_asgn][asgn_no]
                    if not asgn_data[sel_uname][subj_asgn]: del asgn_data[sel_uname][subj_asgn]
                    save_assignment_marks(asgn_data); st.success("Deleted."); st.rerun()
                else: st.warning("Not found.")
            s_asgn = assignment_marks_summary(sel_uname)
            if s_asgn["subjects"]:
                rows_a = [{"Subject":s,"Assignment":an,"Score":sc}
                          for s,v in s_asgn["subjects"].items()
                          for an,sc in v["scores"].items()]
                st.dataframe(pd.DataFrame(rows_a), width='stretch')
                st.info(f"Overall — Avg: {s_asgn['avg']}% | %: {s_asgn['pct']}%")

        with mark_tab3:
            sem_data    = load_semester_marks()
            sem_student = sem_data.get(sel_uname, {})
            subj_sem    = st.text_input("Subject", value="Math", key="sem_subject")

            # Show existing grade for this subject (if any)
            existing_grade = sem_student.get(subj_sem, "")
            existing_label = grade_display_label(existing_grade) if existing_grade else GRADE_OPTIONS[0]
            # Find index in options list
            try:
                default_idx = GRADE_OPTIONS.index(existing_label)
            except ValueError:
                default_idx = 0

            sel_grade_label = st.selectbox(
                "Grade", GRADE_OPTIONS, index=default_idx, key="sem_grade"
            )
            sel_grade = GRADE_LABELS[sel_grade_label]

            cs_save, cs_del = st.columns(2)
            if cs_save.button("Save Semester Grade", key="sem_save"):
                sem_data.setdefault(sel_uname, {})[subj_sem] = sel_grade
                save_semester_marks(sem_data); st.success(f"Saved {subj_sem} → {sel_grade}."); st.rerun()
            if cs_del.button("Delete Subject", key="sem_del"):
                if sel_uname in sem_data and subj_sem in sem_data[sel_uname]:
                    del sem_data[sel_uname][subj_sem]; save_semester_marks(sem_data)
                    st.success("Deleted."); st.rerun()
                else: st.warning("Not found.")

            s_sem = semester_marks_summary(sel_uname)
            if s_sem["subjects"]:
                st.dataframe(pd.DataFrame([
                    {"Subject": s, "Grade": v["grade"]}
                    for s, v in s_sem["subjects"].items()
                ]), width='stretch')

    # ── Download ──
    st.divider()
    st.markdown("#### Download Report")
    d1, d2 = st.columns(2)
    d1.download_button("Download CSV",   data=to_csv(rdf),   file_name="attendance_report.csv",   mime="text/csv")
    d2.download_button("Download Excel", data=to_excel(rdf), file_name="attendance_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ── Full log ──
    with st.expander("📋 View Full Attendance Log"):
        log_df = load_attendance().sort_values("date", ascending=False).reset_index(drop=True)
        if log_df.empty:
            st.info("No records yet.")
        else:
            fc1, fc2 = st.columns(2)
            filter_name = fc1.selectbox("Filter by Student",
                ["All"] + sorted(log_df["name"].unique().tolist()), key="log_filter_name")
            filter_date = fc2.text_input("Filter by Date (YYYY-MM-DD)", key="log_filter_date")
            view = log_df.copy()
            if filter_name != "All": view = view[view["name"] == filter_name]
            if filter_date.strip():  view = view[view["date"] == filter_date.strip()]
            st.dataframe(view.reset_index(drop=True), width='stretch')
            max_row = max(len(log_df)-1, 0)
            del_idx = st.number_input("Row index to delete (0-based)", 0, max_row, step=1, key="log_del_idx")
            if st.button("🗑️ Delete Selected Row", key="log_del_btn"):
                row  = log_df.iloc[int(del_idx)]
                full = load_attendance()
                mask = ((full["username"]==row["username"]) & (full["date"]==row["date"])
                        & (full["status"]==row["status"]))
                if mask.sum() == 0:
                    st.warning("Record not found.")
                else:
                    save_attendance(full[~mask].reset_index(drop=True))
                    st.success(f"Deleted: {row['name']} on {row['date']}.")
                    st.rerun()
            st.download_button("Download Full Log CSV", data=to_csv(log_df),
                               file_name="full_log.csv", mime="text/csv")

# ─── STUDENT DASHBOARD ─────────────────────────────────────────────────────────
def page_dashboard_student():
    st.subheader("My Attendance")
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=5000, key="student_refresh")

    uname = st.session_state.username
    stats = attendance_stats(uname)

    # Top metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Days (since reg.)", stats["total"])
    c2.metric("Present",  stats["present"])
    c3.metric("Absent",   stats["absent"])

    c4, c5, c6 = st.columns(3)
    c4.metric("Attendance %", f"{stats['pct']}%")
    c5.metric("This Week",    stats["weekly"])
    c6.metric("This Month",   stats["monthly"])

    st.caption(f"Tracking from registration date: {stats['reg_date']}")

    cfg       = load_sms_config()
    threshold = int(cfg.get("threshold", 75))
    if stats["pct"] < threshold and stats["total"] > 0:
        st.error(f"Warning: Your attendance is {stats['pct']}% — below the {threshold}% requirement.")

    if stats["total"] > 0:
        fig_pie = px.pie(
            values=[stats["present"], stats["absent"]],
            names=["Present", "Absent"],
            color_discrete_sequence=["#2ecc71" if stats["pct"] >= threshold else "#e74c3c", "#eee"],
            hole=0.6, title=f"Overall: {stats['pct']}%")
        st.plotly_chart(fig_pie, width='stretch')

    df  = load_attendance()
    udf = df[df["username"] == uname].copy()
    if not udf.empty:
        udf["date"] = pd.to_datetime(udf["date"], errors="coerce")
        monthly = udf.groupby(udf["date"].dt.strftime("%b %Y")).size().reset_index()
        monthly.columns = ["Month", "Days Present"]
        st.plotly_chart(px.bar(monthly, x="Month", y="Days Present",
                               title="Monthly Attendance"), width='stretch')
        weekly = udf.groupby(udf["date"].dt.isocalendar().week.astype(str)).size().reset_index()
        weekly.columns = ["Week", "Days Present"]
        st.plotly_chart(px.line(weekly, x="Week", y="Days Present",
                                markers=True, title="Weekly Trend"), width='stretch')
        st.divider()
        st.markdown("#### Download My Report")
        my_rep = udf[["name","date","status"]].rename(
            columns={"name":"Name","date":"Date","status":"Status"})
        r1, r2 = st.columns(2)
        r1.download_button("Download CSV",   data=to_csv(my_rep),
                           file_name=f"{uname}_attendance.csv", mime="text/csv")
        r2.download_button("Download Excel", data=to_excel(my_rep),
                           file_name=f"{uname}_attendance.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ── Marks (read-only) ──
    st.divider()
    st.markdown("#### My Marks")
    m_tab1, m_tab2, m_tab3 = st.tabs(["Internal Marks","Assignment Marks","Semester Marks"])

    with m_tab1:
        s_int = internal_marks_summary(uname)
        if s_int["subjects"]:
            st.dataframe(pd.DataFrame([{"Subject":s,"I1":v["internal1"],"I2":v["internal2"],
                "I3":v["internal3"],"Total":v["total"],"%":v["pct"]}
                for s,v in s_int["subjects"].items()]), width='stretch')
            st.info(f"Overall — Avg: {round(s_int['avg'],1)} | %: {s_int['pct']}%")
        else:
            st.info("No internal marks recorded yet.")

    with m_tab2:
        s_asgn = assignment_marks_summary(uname)
        if s_asgn["subjects"]:
            rows_a = [{"Subject":s,"Assignment":an,"Score":sc}
                      for s,v in s_asgn["subjects"].items()
                      for an,sc in v["scores"].items()]
            st.dataframe(pd.DataFrame(rows_a), width='stretch')
            st.info(f"Overall — Avg: {s_asgn['avg']}% | %: {s_asgn['pct']}%")
        else:
            st.info("No assignment marks recorded yet.")

    with m_tab3:
        s_sem = semester_marks_summary(uname)
        if s_sem["subjects"]:
            st.dataframe(pd.DataFrame([
                {"Subject": s, "Grade": v["grade"]}
                for s, v in s_sem["subjects"].items()
            ]), width='stretch')
        else:
            st.info("No semester marks recorded yet.")

# ─── EDIT ATTENDANCE ───────────────────────────────────────────────────────────
def page_edit_attendance():
    st.subheader("Edit Attendance")
    users    = load_users()
    students = {v["full_name"]: k for k, v in users.items() if v["role"] == "student"}
    if not students:
        st.info("No students registered.")
        return
    name_sel = st.selectbox("Student Name", list(students.keys()))
    uname    = students[name_sel]
    df       = load_attendance()
    st.markdown("##### Existing Records")
    udf = df[df["username"] == uname].sort_values("date", ascending=False)
    if udf.empty:
        st.info("No attendance records yet.")
    else:
        st.dataframe(udf[["name","date","status"]].reset_index(drop=True), width='stretch')
    st.divider()
    sel_date = st.date_input("Select Date", value=date.today())
    ds       = str(sel_date)
    action   = st.radio("Action", ["Add Present","Delete Record"])
    rec_exists = not df[(df["username"] == uname) & (df["date"] == ds)].empty
    if action == "Add Present" and rec_exists:
        st.warning(f"Record already exists for {name_sel} on {ds}.")
    if action == "Delete Record" and not rec_exists:
        st.warning(f"No record found for {name_sel} on {ds}.")
    if st.button("Save"):
        df = load_attendance()
        rec_now = not df[(df["username"] == uname) & (df["date"] == ds)].empty
        if action == "Add Present":
            if rec_now:
                st.warning("Record already exists.")
            else:
                df = pd.concat([df, pd.DataFrame([{"name":name_sel,"username":uname,
                    "date":ds,"status":"Present"}])], ignore_index=True)
                save_attendance(df)
                st.success(f"Attendance added for {name_sel} on {ds}.")
                st.rerun()
        else:
            mask = (df["username"] == uname) & (df["date"] == ds)
            if mask.sum() == 0:
                st.warning("No record found.")
            else:
                save_attendance(df[~mask].reset_index(drop=True))
                st.success(f"Deleted record for {name_sel} on {ds}.")
                st.rerun()

# ─── EDIT CREDENTIALS ──────────────────────────────────────────────────────────
def page_edit_credentials():
    st.subheader("Edit User Credentials")
    users   = load_users()
    targets = {f"{v['full_name']} ({k}) [{v['role']}]": k
               for k, v in users.items() if v["role"] in ("staff","student")}
    if not targets:
        st.info("No staff or students registered.")
        return
    choice   = st.selectbox("Select User", list(targets.keys()))
    uname    = targets[choice]
    new_user = st.text_input("New Username", value=uname)
    new_pass = st.text_input("New Password", type="password")
    if st.button("Save Changes"):
        if new_user != uname and new_user in users:
            st.error("Username already taken.")
            return
        entry = users.pop(uname)
        entry["username"] = new_user
        if new_pass:
            entry["password"] = new_pass
        users[new_user] = entry
        save_users(users)
        st.success("Credentials updated.")

# ─── SMS SETTINGS ──────────────────────────────────────────────────────────────
def page_sms_settings():
    st.subheader("SMS Alert Settings (Twilio)")
    cfg = load_sms_config()
    st.markdown("Get credentials from [twilio.com/console](https://www.twilio.com/console)")
    enabled     = st.toggle("Enable SMS Alerts", value=cfg.get("enabled", False))
    threshold   = st.slider("Low Attendance Threshold (%)", 50, 90,
                            int(cfg.get("threshold", 75)), step=5)
    sid         = st.text_input("Twilio Account SID",  value=cfg.get("sid",""),   type="password")
    token       = st.text_input("Twilio Auth Token",   value=cfg.get("token",""), type="password")
    from_number = st.text_input("Twilio From Number",  value=cfg.get("from_number",""), placeholder="+1XXXXXXXXXX")
    admin_phone = st.text_input("Admin Phone",         value=cfg.get("admin_phone",""), placeholder="+91XXXXXXXXXX")

    c1, c2 = st.columns(2)
    if c1.button("Save Settings"):
        save_sms_config({"enabled":enabled,"threshold":threshold,"sid":sid,"token":token,
                         "from_number":from_number,"admin_phone":admin_phone})
        st.success("SMS settings saved.")
    if c2.button("Send Test SMS to Admin"):
        if not admin_phone:
            st.error("Enter admin phone first.")
        else:
            save_sms_config({"enabled":enabled,"threshold":threshold,"sid":sid,"token":token,
                             "from_number":from_number,"admin_phone":admin_phone})
            ok, err = send_sms_result(admin_phone, "Smart Attendance System: Test SMS successful.")
            st.success(f"Sent to {admin_phone}") if ok else st.error(f"Failed: {err}")

    st.divider()
    st.markdown("#### Manual Alert to Student")
    users    = load_users()
    students = {v["full_name"]: k for k, v in users.items() if v["role"] == "student"}
    if students:
        sel   = st.selectbox("Select Student", list(students.keys()), key="sms_student_sel")
        uname = students[sel]
        stats = attendance_stats(uname)
        phone  = users[uname].get("phone","")
        pphone = users[uname].get("parent_phone","")
        st.info(f"Student: {phone or 'not set'}  |  Parent: {pphone or 'not set'}  |  Attendance: {stats['pct']}%")
        custom_msg = st.text_area("Message", value=(
            f"Dear {sel}, your attendance is {stats['pct']}% "
            f"({stats['present']}/{stats['total']} days). Minimum required is {threshold}%."))
        s1, s2, s3 = st.columns(3)
        if s1.button("Send to Student"):
            ok, err = send_sms_result(phone, custom_msg)
            st.success("Sent.") if ok else st.error(f"Failed: {err}")
        if s2.button("Send to Parent"):
            ok, err = send_sms_result(pphone, f"Parent Alert: {custom_msg}")
            st.success("Sent.") if ok else st.error(f"Failed: {err}")
        if s3.button("Send to Both"):
            ok1, e1 = send_sms_result(phone, custom_msg)
            ok2, e2 = send_sms_result(pphone, f"Parent Alert: {custom_msg}")
            if ok1: st.success("Sent to student.")
            else:   st.error(f"Student failed: {e1}")
            if ok2: st.success("Sent to parent.")
            else:   st.error(f"Parent failed: {e2}")

    st.divider()
    st.markdown("#### Staff Phone Numbers")
    st.caption("Staff need a phone number to receive low-attendance alerts.")
    staff_users = {k: v for k, v in users.items() if v.get("role") == "staff"}
    if not staff_users:
        st.info("No staff registered.")
    else:
        for suname, sdata in staff_users.items():
            cur_phone = sdata.get("phone", "")
            new_phone = st.text_input(
                f"{sdata['full_name']} ({suname})",
                value=cur_phone,
                placeholder="+91XXXXXXXXXX",
                key=f"staff_phone_{suname}"
            )
            if new_phone != cur_phone:
                if st.button(f"Save phone for {sdata['full_name']}", key=f"save_sp_{suname}"):
                    users[suname]["phone"] = new_phone.strip()
                    save_users(users)
                    st.success(f"Phone updated for {sdata['full_name']}.")
                    st.rerun()

    st.divider()
    st.markdown(f"#### Bulk Alert — Students Below {threshold}%")

    # Show which staff will receive alerts
    staff_with_phone    = [(v["full_name"], v["phone"]) for v in users.values()
                           if v.get("role") == "staff" and v.get("phone","").strip()]
    staff_without_phone = [v["full_name"] for v in users.values()
                           if v.get("role") == "staff" and not v.get("phone","").strip()]
    if staff_with_phone:
        st.success(f"Staff who will receive alerts: {', '.join(n for n,_ in staff_with_phone)}")
    if staff_without_phone:
        st.warning(f"Staff with no phone (will NOT receive alerts): {', '.join(staff_without_phone)}"
                   " — set their phone above.")

    if st.button("Send Alerts to All Low-Attendance Students"):
        sent, failed = 0, 0
        staff_phones = [v["phone"].strip() for v in users.values()
                        if v.get("role") == "staff" and v.get("phone","").strip()]
        adm_phone    = load_sms_config().get("admin_phone", "")
        for uname, udata in users.items():
            if udata.get("role") != "student":
                continue
            stats = attendance_stats(uname)
            if stats["total"] > 0 and stats["pct"] < threshold:
                msg   = (f"Dear {udata['full_name']}, your attendance is {stats['pct']}% "
                         f"({stats['present']}/{stats['total']} days). "
                         f"Minimum required is {threshold}%.")
                alert = (f"Low Attendance Alert: {udata['full_name']} is at {stats['pct']}% "
                         f"({stats['present']}/{stats['total']} days).")
                ok1, _ = send_sms_result(udata.get("phone", ""), msg)
                ok2, _ = send_sms_result(udata.get("parent_phone", ""), f"Parent Alert: {alert}")
                if adm_phone:
                    send_sms_result(adm_phone, alert)
                for sp in staff_phones:
                    send_sms_result(sp, alert)
                sent   += 1 if (ok1 or ok2) else 0
                failed += 1 if not (ok1 or ok2) else 0
        st.success(f"Alerts sent for {sent} student(s). Failed: {failed}.")

# ─── HOLIDAY MANAGEMENT PAGE ──────────────────────────────────────────────────
def page_holidays():
    st.subheader("Holiday Management")
    holidays = load_holidays()

    st.markdown("#### Current Holidays")
    if holidays:
        hdf = pd.DataFrame(sorted(holidays), columns=["Date"])
        hdf["Day"] = hdf["Date"].apply(lambda d: d.strftime("%A"))
        st.dataframe(hdf, width='stretch')
    else:
        st.info("No holidays added yet.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Add Holiday")
        new_hdate = st.date_input("Select Holiday Date", key="add_hdate")
        hlabel    = st.text_input("Label (optional)", key="add_hlabel")
        if st.button("Add Holiday"):
            holidays.add(new_hdate)
            save_holidays(holidays)
            st.success(f"Added {new_hdate} ({new_hdate.strftime('%A')}) as holiday.")
            st.rerun()

    with col2:
        st.markdown("#### Remove Holiday")
        if holidays:
            del_choice = st.selectbox("Select to remove",
                                      sorted(str(d) for d in holidays), key="del_hdate")
            if st.button("Remove Holiday"):
                holidays.discard(date.fromisoformat(del_choice))
                save_holidays(holidays)
                st.success(f"Removed {del_choice}.")
                st.rerun()
        else:
            st.info("Nothing to remove.")

    st.divider()
    st.markdown("#### Working Days Preview")
    users    = load_users()
    students = [u for u, v in users.items() if v["role"] == "student"]
    if students:
        sel_uname = st.selectbox("Student", students,
                                 format_func=lambda u: users[u]["full_name"],
                                 key="hol_preview_sel")
        reg_str = users[sel_uname].get("reg_date", str(date.today()))
        reg_d   = date.fromisoformat(reg_str)
        wdays   = working_days_between(reg_d, date.today(), holidays)
        st.info(f"Working days from {reg_str} to today: **{len(wdays)}**  "
                f"(excl. {sum(1 for d in wdays if d in holidays)} holidays, weekends auto-excluded)")

# ─── MAIN ROUTER ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Attendance System", layout="wide")

if not st.session_state.logged_in:
    page_login()
else:
    role = st.session_state.role
    name = st.session_state.full_name

    with st.sidebar:
        st.title("Smart Attendance")
        st.write(f"Logged in as: **{name}** ({role})")
        st.divider()
        if role == "admin":
            menu = st.radio("Menu", ["Dashboard","Register User","Capture Face",
                                     "Live Attendance","Edit Attendance",
                                     "Edit Credentials","Holidays","SMS Settings","Logout"])
        elif role == "staff":
            menu = st.radio("Menu", ["Dashboard","Capture Face","Live Attendance",
                                     "Edit Attendance","Edit Credentials","Logout"])
        else:
            menu = st.radio("Menu", ["My Attendance","Logout"])

    if menu == "Logout":
        for k in ["logged_in","username","role","full_name"]:
            st.session_state[k] = "" if k != "logged_in" else False
        st.rerun()
    elif menu == "Dashboard":
        page_dashboard_admin_staff()
    elif menu == "My Attendance":
        page_dashboard_student()
    elif menu == "Register User":
        reg_role = st.selectbox("Register as", ["admin","staff","student"])
        page_register(reg_role)
    elif menu == "Capture Face":
        page_face_capture()
    elif menu == "Live Attendance":
        page_live_attendance()
    elif menu == "Edit Attendance":
        page_edit_attendance()
    elif menu == "Edit Credentials":
        page_edit_credentials()
    elif menu == "Holidays":
        page_holidays()
    elif menu == "SMS Settings":
        page_sms_settings()
