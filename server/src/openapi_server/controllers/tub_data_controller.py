import connexion
import six

from openapi_server.models.tub_data_point import TubDataPoint  # noqa: E501
from openapi_server import util
from openapi_server.data_access.data import get_image_path, load_tub_data, load_tub_data_by_id


def get_tub_data_points(car_id, tub_id):  # noqa: E501
    """Returns a single data point in a tub

      # noqa: E501

    :param car_id: ID of car to return
    :type car_id: str
    :param tub_id: ID of tub to return
    :type tub_id: str

    :rtype: TubDataPoint[]
    """

    # Prevent directory traversal
    if ".." in car_id:
        return "Bad Request", 400
    if ".." in tub_id:
        return "Bad Request", 400

    data = load_tub_data(car_id, tub_id)
    if data is None:
        return "Not Found", 404

    return data


def get_image_by_car_and_tub_and_id(car_id, tub_id, image_id):  # noqa: E501
    """Returns an image for single datapoint

      # noqa: E501

    :param car_id: ID of car to return
    :type car_id: str
    :param tub_id: ID of tub to return
    :type tub_id: str
    :param imageId: ID of image to return
    :type imageId: str

    :rtype: None
    """
    # Prevent directory traversal
    if ".." in car_id:
        return "Bad Request", 400
    if ".." in tub_id:
        return "Bad Request", 400

    image_path = get_image_path(car_id, tub_id, image_id)
    if image_path is None:
        return "Not Found", 404

    with open(image_path, "rb") as f:
        return f.read()


def get_tub_data_point_by_car_and_id(car_id, tub_id, data_id):  # noqa: E501
    """Returns a single data point in a tub

      # noqa: E501

    :param car_id: ID of car to return
    :type car_id: str
    :param tub_id: ID of tub to return
    :type tub_id: str
    :param data_id: ID of tub to return
    :type data_id:

    :rtype: TubDataPoint
    """

    # Prevent directory traversal
    if ".." in car_id:
        return "Bad Request", 400
    if ".." in tub_id:
        return "Bad Request", 400

    index = -1
    try:
        index = int(data_id)
    except ValueError:
        return "Bad Request", 400

    data = load_tub_data_by_id(car_id, tub_id, index)
    if data is None:
        return "Not Found", 404

    return data
