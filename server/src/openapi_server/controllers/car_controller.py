import connexion
import six

from openapi_server.models.car import Car  # noqa: E501
from openapi_server.data_access.data import load_cars, load_car_by_id, load_tubs
from openapi_server import util


def get_car_by_id(car_id):  # noqa: E501
    """Find car by ID

    Returns a single car # noqa: E501

    :param car_id: ID of car to return
    :type car_id: str

    :rtype: Car
    """

    # Prevent directory traversal
    if ".." in car_id:
        return "Bad Request", 400

    car = load_car_by_id(car_id)
    if car is None:
        return "Not Found", 404

    return Car(car.id, car.name)


def get_cars():  # noqa: E501
    """List all cars

    Multiple status values can be provided with comma separated strings # noqa: E501


    :rtype: List[Car]
    """
    return load_cars()
