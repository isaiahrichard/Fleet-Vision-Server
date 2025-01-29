from flask import Blueprint, request, make_response

obd = Blueprint("obd", __name__)


@obd.route("/obd", methods=["GET"])
def get_obd():
    # Get data from shared queue with TCP server
    # data = tcp_server.data_queue.get()

    return make_response("Request for OBD", 200)
