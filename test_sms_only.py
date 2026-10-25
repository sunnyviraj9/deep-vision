"""Test SMS only — run this on mobile hotspot"""
from twilio.rest import Client

SID   = "ACe7c23e091e8287cbba81d1b7b9c4a0ee"
TOKEN = "0edc51fcb4c0b582c70c1567fd87046c"
FROM  = "+12694481785"
TO    = "+916300419322"

print(f"Sending SMS to {TO}...", end=" ", flush=True)
try:
    msg = Client(SID, TOKEN).messages.create(
        body="DeepVision Crowd Monitor: SMS alert is working!", from_=FROM, to=TO)
    print(f"✅ SUCCESS  SID={msg.sid}  Status={msg.status}")
except Exception as e:
    print(f"❌ {type(e).__name__}: {e}")
