#!/usr/bin/env python3

from openapi_server import encoder
import connexion
from flask_cors import CORS
# from os.path import dirname, realpath, join
# dir_path = dirname(realpath(__file__))
# lal = join(dir_path, "./openapi/")
# print(lal)


def main():
    app = connexion.App(__name__, specification_dir="./openapi/")
    # app = connexion.App(__name__, specification_dir=lal)
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('openapi.yaml',
                arguments={'title': 'Markku portal API'},
                pythonic_params=True)
    CORS(app.app)
    app.run(port=8080)


if __name__ == '__main__':
    main()
