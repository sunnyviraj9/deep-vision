"""Test only SMS via Twilio (uses HTTPS port 443, rarely blocked)"""
TWILIO_SID   = "ACe7c23e091e8287cbba81d1b7b9c4a0ee"
TWILIO_TOKEN = "9244ccfaaffa0cf302816b44569a6c1b"
TWILIO_FROM  = "+12694481785"
ALERT_PHONE  = "+916300419322"

print("Testing SMS via Twilio...", end=" ", flush=True)
try:
    from twilio.rest import Client
    client = Client(TWILIO_SID, TWILIO_TOKEN)
    msg = client.messages.create(
        body="DeepVision Crowd Monitor: SMS alert test successful!",
        from_=TWILIO_FROM,
        to=ALERT_PHONE
    )
    print(f"SUCCESS  SID={msg.sid}  Status={msg.status}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
