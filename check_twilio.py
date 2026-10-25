"""Check what phone numbers and messaging services are on this Twilio account"""
from twilio.rest import Client

SID   = "ACe7c23e091e8287cbba81d1b7b9c4a0ee"
TOKEN = "0edc51fcb4c0b582c70c1567fd87046c"
TO    = "+916300419322"

client = Client(SID, TOKEN)

print("── Phone Numbers on account ──────────────────")
for num in client.incoming_phone_numbers.list():
    print(f"  {num.phone_number}  ({num.friendly_name})  capabilities={num.capabilities}")

print("\n── Messaging Services ────────────────────────")
for svc in client.messaging.v1.services.list():
    print(f"  SID={svc.sid}  Name={svc.friendly_name}")

print("\n── Trying alphanumeric sender (works for India) ──")
try:
    msg = client.messages.create(
        body="DeepVision Alert: SMS test from India sender",
        from_="DeepVision",   # alphanumeric sender ID — works in India
        to=TO
    )
    print(f"  ✅ Sent! SID={msg.sid}  Status={msg.status}")
except Exception as e:
    print(f"  ❌ {type(e).__name__}: {e}")
