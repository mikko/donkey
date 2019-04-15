# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from openapi_server.models.base_model_ import Model
from openapi_server import util


class TrainingInfo(Model):
    """NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).

    Do not edit the class manually.
    """

    def __init__(self, status=None):  # noqa: E501
        """TrainingInfo - a model defined in OpenAPI

        :param status: The status of this TrainingInfo.  # noqa: E501
        :type status: str
        """
        self.openapi_types = {
            'status': str
        }

        self.attribute_map = {
            'status': 'status'
        }

        self._status = status

    @classmethod
    def from_dict(cls, dikt) -> 'TrainingInfo':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The TrainingInfo of this TrainingInfo.  # noqa: E501
        :rtype: TrainingInfo
        """
        return util.deserialize_model(dikt, cls)

    @property
    def status(self):
        """Gets the status of this TrainingInfo.


        :return: The status of this TrainingInfo.
        :rtype: str
        """
        return self._status

    @status.setter
    def status(self, status):
        """Sets the status of this TrainingInfo.


        :param status: The status of this TrainingInfo.
        :type status: str
        """
        if status is None:
            raise ValueError("Invalid value for `status`, must not be `None`")  # noqa: E501

        self._status = status
