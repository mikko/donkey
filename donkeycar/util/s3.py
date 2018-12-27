import os
import shutil
import boto3

def archive_tub(path):
  base_name = os.path.splitext(path)[0]
  return shutil.make_archive(base_name, 'zip', path)

def upload(path):
  s3 = boto3.resource('s3')
  filename = os.path.basename(path)
  s3.meta.client.upload_file(path, 'maximum-aittack-tubs', filename)
