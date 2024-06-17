from flask import Blueprint, request, make_response
from db_collections import model_col
import requests

model = Blueprint("model", __name__)


@model.route("/model", methods=["GET"])
def get_obd():
    return make_response("Request for model", 200)
