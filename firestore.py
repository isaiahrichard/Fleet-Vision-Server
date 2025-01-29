from firebase_admin import credentials, firestore, initialize_app

cred = credentials.Certificate("key.json")
default_app = initialize_app(cred)
db = firestore.client()
drive_sessions = db.collection("drive_sessions")
