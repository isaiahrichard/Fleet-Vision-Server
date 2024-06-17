from flask import Blueprint, request, make_response
from db_collections import obd_col
import requests

obd = Blueprint("obd", __name__)


@obd.route("/", methods=["GET"])
def get_obd():
    return make_response("OK", 200)
