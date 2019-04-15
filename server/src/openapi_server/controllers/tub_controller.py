import connexion
import six

from datetime import datetime
from openapi_server.models.tub import Tub  # noqa: E501
from openapi_server import util
from openapi_server.data_access.data import load_tubs


def get_tub_by_car_and_id(car_id, tubId):  # noqa: E501
    """Returns data for a car&#39;s tub

     # noqa: E501

    :param car_id: ID of car to return
    :type car_id: int
    :param tubId: ID of tub to return
    :type tubId: str

    :rtype: TubResponse
    """
    tub_data = load_tub_data(car_id, tubId)
    if tub_data is None:
        return "Not Found", 404

    return None
#     return TubResponse(tubId, datetime.now(), datetime.now(), tub_data)


def get_tubs_by_car(car_id):  # noqa: E501
    """Returns all tubs for a car

     # noqa: E501

    :param car_id: ID of car to return
    :type car_id: int

    :rtype: Tub
    """

    # Prevent directory traversal
    if ".." in car_id:
        return "Bad Request", 400

    tubs = load_tubs(car_id)
    if tubs is None:
        return "Not Found", 404

    return tubs
