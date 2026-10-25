import requests, sys
print("Testing Twilio via requests...")
try:
    r = requests.post(
        f"https://api.twilio.com/2010-04-01/Accounts/ACe7c23e091e8287cbba81d1b7b9c4a0ee/Messages.json",
        auth=("ACe7c23e091e8287cbba81d1b7b9c4a0ee", "9244ccfaaffa0cf302816b44569a6c1b"),
        data={
            "From": "+12694481785",
            "To":   "+916300419322",
            "Body": "DeepVision test alert!"
        },
        timeout=15
    )
    print(f"Status: {r.status_code}")
    print(r.text[:500])
except requests.exceptions.SSLError as e:
    print(f"SSL Error (network proxy blocking): {e}")
except requests.exceptions.Timeout:
    print("Timeout after 15s")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
