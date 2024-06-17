from flask import Blueprint, request, make_response
from db_collections import obd_col
import requests

obd = Blueprint("obd", __name__)


@obd.route("/obd", methods=["GET"])
def get_obd():
    return make_response("Request for OBD", 200)
