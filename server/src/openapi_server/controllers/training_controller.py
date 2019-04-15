import connexion
import six
from multiprocessing import Process

from openapi_server.models.train_command import TrainCommand  # noqa: E501
from openapi_server.models.training_info import TrainingInfo  # noqa: E501
from openapi_server import util

print_process = None


def get_training_info_by_car_id(car_id):  # noqa: E501
    """Returns information about current training session

     # noqa: E501

    :param car_id: ID of car to return
    :type car_id: str

    :rtype: TrainingInfo
    """
    return TrainingInfo("stopped")


def handle_training(car_id, train_command=None):  # noqa: E501
    """Change car&#39;s training session

     # noqa: E501

    :param car_id: ID of car to return
    :type car_id: str
    :param train_command: Command to send
    :type train_command: dict | bytes

    :rtype: TrainingInfo
    """
    if connexion.request.is_json:
        train_command = TrainCommand.from_dict(connexion.request.get_json())  # noqa: E501

    global print_process
    # instantiating without any argument
    print_process = Process(target=train_func)
    print('Training start')
    print_process.start()
    return TrainingInfo("started")


def train_func():
    train(tub_path, 'api_model')
