#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.33  # yolox-x
        self.width = 1.25  # yolox-x
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
