# coding: utf-8

from __future__ import absolute_import
import unittest

from flask import json
from six import BytesIO

from openapi_server.models.tub_data_point import TubDataPoint  # noqa: E501
from openapi_server.test import BaseTestCase


class TestTubDataController(BaseTestCase):
    """TubDataController integration test stubs"""

    def test_get_image_by_car_and_tub_and_id(self):
        """Test case for get_image_by_car_and_tub_and_id

        Returns an image for single datapoint
        """
        headers = {
        }
        response = self.client.open(
            '/car/{car_id}/tub/{tub_id}/image/{image_id}'.format(
                car_id='car_id_example', tub_id='tub_id_example', image_id='image_id_example'),
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_tub_data_point_by_car_and_id(self):
        """Test case for get_tub_data_point_by_car_and_id

        Returns a single data point in a tub
        """
        headers = {
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/car/{car_id}/tub/{tub_id}/data/{data_id}'.format(
                car_id='car_id_example', tub_id='tub_id_example', data_id='data_id_example'),
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_tub_data_points(self):
        """Test case for get_tub_data_points

        Returns all data points in a tub
        """
        headers = {
            'Accept': 'application/json',
        }
        response = self.client.open(
            '/car/{car_id}/tub/{tub_id}/data'.format(
                car_id='car_id_example', tub_id='tub_id_example'),
            method='GET',
            headers=headers)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    unittest.main()
