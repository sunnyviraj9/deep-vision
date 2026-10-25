"""
test_alerts.py — Quick test for email + SMS
Run:  python test_alerts.py
"""
import smtplib, socket
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

EMAIL_USER   = "sunnyviraj9@gmail.com"
EMAIL_PASS   = "qdpdxivnnvfiqexr"       # new app password
ALERT_EMAIL  = "sunnyviraj9@gmail.com"

TWILIO_SID   = "ACe7c23e091e8287cbba81d1b7b9c4a0ee"
TWILIO_TOKEN = "0edc51fcb4c0b582c70c1567fd87046c"
TWILIO_FROM  = "+12694481785"
ALERT_PHONE  = "+916300419322"

# ── Email ─────────────────────────────────────────────────────────────────────
print("── Email ──────────────────────────────────")
for port, method in [(587, "STARTTLS"), (465, "SSL")]:
    print(f"  Trying port {port} ({method})...", end=" ", flush=True)
    try:
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_USER
        msg["To"]      = ALERT_EMAIL
        msg["Subject"] = f"DeepVision Alert Test ({method})"
        msg.attach(MIMEText(f"Test via port {port} {method} — alert system working!", "plain"))
        if method == "STARTTLS":
            with smtplib.SMTP("smtp.gmail.com", port, timeout=10) as smtp:
                smtp.ehlo(); smtp.starttls(); smtp.login(EMAIL_USER, EMAIL_PASS)
                smtp.sendmail(EMAIL_USER, ALERT_EMAIL, msg.as_string())
        else:
            with smtplib.SMTP_SSL("smtp.gmail.com", port, timeout=10) as smtp:
                smtp.login(EMAIL_USER, EMAIL_PASS)
                smtp.sendmail(EMAIL_USER, ALERT_EMAIL, msg.as_string())
        print("✅ SUCCESS — check your inbox!")
        break
    except smtplib.SMTPAuthenticationError:
        print("❌ Wrong password — re-check app password")
        break
    except socket.timeout:
        print(f"❌ Timed out on port {port}")
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")

# ── SMS ──────────────────────────────────────────────────────────────────────
print("\n── SMS ─────────────────────────────────────")
print(f"  Sending to {ALERT_PHONE}...", end=" ", flush=True)
try:
    from twilio.rest import Client
    sms = Client(TWILIO_SID, TWILIO_TOKEN).messages.create(
        body="DeepVision Crowd Monitor: test alert — SMS working!",
        from_=TWILIO_FROM, to=ALERT_PHONE
    )
    print(f"✅ SUCCESS — SID: {sms.sid}")
except Exception as e:
    print(f"❌ {type(e).__name__}: {e}")
