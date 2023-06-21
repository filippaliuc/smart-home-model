import pyrebase
from firebaseConfig import firebaseConfig

firebase = pyrebase.initialize_app(firebaseConfig)

storage = firebase.storage()
database = firebase.database()

