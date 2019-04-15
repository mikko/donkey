# coding: utf-8

from __future__ import absolute_import
import unittest

from flask import json
from six import BytesIO

from openapi_server.models.one_of_start_command_stop_command import OneOfStartCommandStopCommand  # noqa: E501
from openapi_server.models.training_info import TrainingInfo  # noqa: E501
from openapi_server.models.unknownbasetype import UNKNOWN_BASE_TYPE  # noqa: E501
from openapi_server.test import BaseTestCase


class TestTrainingController(BaseTestCase):
    """TrainingController integration test stubs"""

    def test_get_training_info_by_car_id(self):
        """Test case for get_training_info_by_car_id

        Returns information about current training session
        """
        headers = {
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/car/{car_id}/training'.format(car_id='car_id_example'),
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_handle_training(self):
        """Test case for handle_training

        Change car's training session
        """
        unknown_base_type = {}
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        response = self.client.open(
            '/car/{car_id}/training'.format(car_id='car_id_example'),
            method='POST',
            headers=headers,
            data=json.dumps(unknown_base_type),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
