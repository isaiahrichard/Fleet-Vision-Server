from flask import Flask
from routes.obd import obd
from routes.model import model
import threading
import tcp_server

app = Flask(__name__)
app.register_blueprint(obd)
app.register_blueprint(model)

if __name__ == "__main__":
    # start TCP and Flask server on separate threads
    # tcp_thread = threading.Thread(target=tcp_server.start_tcp_server)
    # tcp_thread.start()
    
    app.run(debug=True)
