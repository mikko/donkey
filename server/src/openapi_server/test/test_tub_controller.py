# coding: utf-8

from __future__ import absolute_import
import unittest

from flask import json
from six import BytesIO

from openapi_server.models.tub import Tub  # noqa: E501
from openapi_server.test import BaseTestCase


class TestTubController(BaseTestCase):
    """TubController integration test stubs"""

    def test_get_tub_by_car_and_id(self):
        """Test case for get_tub_by_car_and_id

        Returns a single tub for a car
        """
        headers = {
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/car/{car_id}/tub/{tub_id}'.format(
                car_id='car_id_example', tub_id='tub_id_example'),
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_tubs_by_car(self):
        """Test case for get_tubs_by_car

        Returns all tubs for a car
        """
        headers = {
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/car/{car_id}/tub'.format(car_id='car_id_example'),
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
