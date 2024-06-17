from flask import Blueprint, request, make_response
from db_collections import model_col
import requests

model = Blueprint("model", __name__)


@model.route("/", methods=["GET"])
def get_obd():
    return make_response("OK", 200)
