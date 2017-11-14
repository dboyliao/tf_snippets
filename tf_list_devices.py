#!/usr/bin/env python3
# -*- coding:utf8 -*-
# pylint: disable=E0611,C0103
from tensorflow.python.client import device_lib
# pylint: enable=E0611

devices = device_lib.list_local_devices()
dev = devices[0]
print(type(dev))
print(dev.device_type)  # CPU or GPU
print(dev.memory_limit)
print(dev.name)
print(dev.physical_device_desc)
