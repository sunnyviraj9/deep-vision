# ===============================
# DeepVision Crowd Monitor
# Milestone 4: Dashboard + Email Alerts
# ===============================
import os, smtplib, io
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
IMG_SZ          = (256, 256)
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALERT_THRESHOLD = int(os.getenv("ALERT_THRESHOLD", "500"))
_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(_SCRIPT_DIR, "tiny_mcnn.pth")

# -------------------------------
# Email credentials
# -------------------------------
EMAIL_USER  = "sunnyviraj9@gmail.com"
EMAIL_PASS  = "qdpdxivnnvfiqexr"
ALERT_EMAIL = "sunnyviraj9@gmail.com"

# -------------------------------
# Model: Tiny-MCNN
# -------------------------------
class TinyMCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64,  1, 1)
        )

    def forward(self, x):
        o = self.net(x)
        if o.shape[2:] != x.shape[2:]:
            o = F.interpolate(o, size=(x.shape[2], x.shape[3]),
                              mode='bilinear', align_corners=False)
        return o

# -------------------------------
# Load model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    m = TinyMCNN().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        m.eval()
        return m, True
    m.eval()
    return m, False

# -------------------------------
# Inference
# -------------------------------
def predict_count(img: Image.Image, model):
    tf  = transforms.ToTensor()
    arr = np.array(img.resize(IMG_SZ)).astype("float32") / 255.0
    t   = tf(arr).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        den = model(t).cpu().squeeze().numpy()
    return float(den.sum()), den

# -------------------------------
# Email alert
# -------------------------------
def send_email(count):
    try:
        subject = "🚨 DeepVision Crowd Alert"
        body    = (
            f"OVERCROWDING DETECTED\n\n"
            f"Estimated crowd count : {count:.0f}\n"
            f"Alert threshold       : {ALERT_THRESHOLD}\n\n"
            f"Please take immediate action.\n\n"
            f"-- DeepVision Crowd Monitor"
        )
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_USER
        msg["To"]      = ALERT_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Try port 587 (STARTTLS), fall back to port 465 (SSL)
        sent = False
        for port, use_tls in [(587, True), (465, False)]:
            try:
                if use_tls:
                    with smtplib.SMTP("smtp.gmail.com", port, timeout=10) as s:
                        s.ehlo(); s.starttls(); s.login(EMAIL_USER, EMAIL_PASS)
                        s.sendmail(EMAIL_USER, ALERT_EMAIL, msg.as_string())
                else:
                    with smtplib.SMTP_SSL("smtp.gmail.com", port, timeout=10) as s:
                        s.login(EMAIL_USER, EMAIL_PASS)
                        s.sendmail(EMAIL_USER, ALERT_EMAIL, msg.as_string())
                sent = True
                break
            except smtplib.SMTPAuthenticationError:
                st.error("📧 Email failed: wrong Gmail app password.")
                return
            except Exception:
                continue   # try next port

        if sent:
            st.success(f"📧 Email alert sent to **{ALERT_EMAIL}**")
        else:
            st.error("� Email failed: could not reach Gmail. Check your network.")
    except Exception as e:
        st.error(f"� Email error: {e}")

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="DeepVision Crowd Monitor", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg, #111111, #000000); color: white; }
    .stButton>button {
        background-color: #FFD700 !important; color: black !important;
        font-weight: bold !important; border-radius: 12px !important;
        padding: 0.6em 1.4em !important; border: none !important;
        font-size: 1rem !important;
    }
    .stButton>button:hover { background-color: #FFC300 !important; transform: scale(1.04); }
    h1, h2, h3, h4 { color: #FFD700 !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("🧑‍🤝‍🧑 DeepVision Crowd Monitor")
st.caption("AI-powered crowd density estimation and overcrowding detection")

model, model_loaded = load_model()
if model_loaded:
    st.caption(f"✅ Trained model loaded  |  device: {DEVICE}  |  alert threshold: {ALERT_THRESHOLD}")
else:
    st.caption(f"⚠️ No trained model found — run `python train.py` first.")

st.divider()

uploaded = st.file_uploader("📤 Upload a crowd image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col_img, col_heat = st.columns(2)
    with col_img:
        st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    count, density = predict_count(img, model)

    with col_heat:
        fig, ax = plt.subplots(facecolor="black")
        ax.imshow(density, cmap="jet")
        ax.set_title("Predicted Density Map", color="white")
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor="black", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, caption="🌡️ Density Heatmap", use_container_width=True)
        plt.close(fig)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Estimated Crowd Count", f"{count:.0f}")
    with c2:
        st.metric("Status", "⚠️ High Crowd Density!" if count > ALERT_THRESHOLD else "✅ Normal")

    if count > ALERT_THRESHOLD:
        st.error(f"🚨 Count **{count:.0f}** exceeds threshold **{ALERT_THRESHOLD}**")
        if st.button("🚨 Send Email Alert"):
            send_email(count)
