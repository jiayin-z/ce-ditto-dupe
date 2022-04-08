# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Site level customizations.

This script is imported by python when any python script is run.
https://docs.python.org/2/library/site.html.
This script configures python logging to write stderr and stdout to a
rotating log file.
"""


import json
import logging
import logging.handlers
import os
import sys
import traceback

from pythonjsonlogger import jsonlogger

try:
  import google3  # pylint: disable=g-import-not-at-top
  _RUNNING_INTERNAL = True
except ImportError:
  _RUNNING_INTERNAL = False

# The file path the last error stacktrace is stored.
TRACEBACK_FILE_PATH = '/tmp/last_traceback.log'


def exception_handler(exctype, value, tb):
  """Handler for uncaught exceptions.

  We want to log uncaught exceptions using the logger so that we properly
  format it for export to Cloud Logging.

  Args:
    exctype: Type of the exception.
    value: The exception.
    tb: The traceback.
  """
  logging.exception(''.join(traceback.format_exception(exctype, value, tb)))


class CustomJsonFormatter(jsonlogger.JsonFormatter):
  """Formats log lines in JSON.
  """

  def _replace_with_original(self, log_record, key):
    """Replaces log_record[key] with log_record['original_' + key]."""
    original_key = 'original_%s' % key
    if original_key in log_record:
      log_record[key] = log_record[original_key]
      del log_record[original_key]

  def process_log_record(self, log_record):
    """Modifies fields in the log_record to match fluentd's expectations."""
    # Support overrides from runcloudml.py.
    self._replace_with_original(log_record, 'created')
    self._replace_with_original(log_record, 'pathname')
    self._replace_with_original(log_record, 'lineno')
    self._replace_with_original(log_record, 'thread')

    log_record['severity'] = log_record['levelname']
    log_record['timestampSeconds'] = int(log_record['created'])
    log_record['timestampNanos'] = int(
        (log_record['created'] % 1) * 1000 * 1000 * 1000)

    return log_record


def get_log_level():
  """Reads the log level from environment variable CLOUD_ML_LOG_LEVEL."""
  log_level = logging.INFO
  log_level_env_variable = 'CLOUD_ML_LOG_LEVEL'
  if log_level_env_variable in os.environ:
    log_level_name = os.environ[log_level_env_variable]
    if log_level_name == 'DEBUG':
      log_level = logging.DEBUG
    elif log_level_name == 'WARNING':
      log_level = logging.WARNING
    elif log_level_name == 'ERROR':
      log_level = logging.ERROR
    elif log_level_name == 'CRITICAL':
      log_level = logging.CRITICAL
  return log_level


class FilterLogs(object):
  """Filter used to remove noisy logs.
  """

  def filter(self, record):
    """Returns True for records that should be logged."""
    if (record.pathname.endswith('googleapiclient/discovery.py') and
        record.levelname == 'INFO'):
      # The discovery client log messages are noisy, as several are logged
      # for each api call we make.
      return False
    try:
      if str(record.msg).find('Traceback (most recent call last)') == 0:
        traceback_path = os.environ.get('TRACEBACK_FILE_PATH',
                                        TRACEBACK_FILE_PATH)
        with open(traceback_path, 'w') as traceback_log:
          traceback_log.write(record.msg)
    except Exception:  # pylint: disable=broad-except
      pass
    return True


def is_ucaip_job():
  """Determine if we are running in the uCAIP environment."""
  cluster_spec = os.environ.get('CLUSTER_SPEC', None)
  if not cluster_spec:
    # CLUSTER_SPEC is provided for all CAIP jobs, but not for uCAIP single
    # replica jobs.
    return True

  cluster = json.loads(cluster_spec).get('cluster', None)
  if not cluster:
    return True

  # uCAIP jobs have an endpoint in the format of '...-workerpoolx-...'
  # 'workerpool0' is always set in CLUSTER_SPEC for uCAIP jobs.
  chief_endpoints = cluster.get('workerpool0', None)
  return chief_endpoints and ('workerpool' in chief_endpoints[0])


def configure_ucaip_logger():
  """Configures logging to write JSON log entries to a rotating file."""
  formatter = CustomJsonFormatter(
      '%(name)s|%(levelname)s|%(message)s|%(created)f'
      '|%(lineno)d|%(pathname)s', '%Y-%m-%dT%H:%M:%S')
  file_handler = logging.handlers.RotatingFileHandler(
      os.environ['LOG_FILE_TO_WRITE'],
      maxBytes=(200 * 1024 * 1024),
      backupCount=2)
  file_handler.addFilter(FilterLogs())
  file_handler.setFormatter(formatter)

  # Configure the root logger, ensuring that we always have at least one
  # handler.
  root_logger = logging.getLogger()
  root_logger_previous_handlers = list(root_logger.handlers)
  root_logger.addHandler(file_handler)

  for h in root_logger_previous_handlers:
    root_logger.removeHandler(h)
  root_logger.setLevel(get_log_level())

  # Insert exception handler for uncaught exceptions.
  sys.excepthook = exception_handler


def configure_caip_logger():
  """For CAIP, only add the log filter to ensure we write tracebacks."""
  root_logger = logging.getLogger()
  root_logger.addFilter(FilterLogs())

  # Insert exception handler for uncaught exceptions.
  sys.excepthook = exception_handler


def _customize():
  if is_ucaip_job():
    configure_ucaip_logger()
  else:
    configure_caip_logger()


_customize()