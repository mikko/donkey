# coding: utf-8

from __future__ import absolute_import
import unittest

from flask import json
from six import BytesIO

from openapi_server.models.car import Car  # noqa: E501
from openapi_server.test import BaseTestCase


class TestCarController(BaseTestCase):
    """CarController integration test stubs"""

    def test_get_car_by_id(self):
        """Test case for get_car_by_id

        Find car by ID
        """
        headers = {
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/car/{car_id}'.format(car_id='car_id_example'),
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_cars(self):
        """Test case for get_cars

        List all cars
        """
        headers = {
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/car',
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
