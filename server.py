from flask import Flask
from routes.obd import obd
from routes.model import model

app = Flask(__name__)
app.register_blueprint(obd)
app.register_blueprint(model)


if __name__ == "__main__":
    app.run(debug=True)
