from pymongo import MongoClient

# add client string
db = MongoClient()

# add db name
CapstoneDB = db.get_database("")

obd_col = CapstoneDB.get_collection("obd")
model_col = CapstoneDB.get_collection("model")
