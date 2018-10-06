#! /usr/bin/python3

from netdef_slim.tensorflow.controller.base_controller import BaseTFController
from netdef_slim.tensorflow.controller.net_actions import NetActions
import os

class Controller(BaseTFController):
    base_path = os.path.dirname(os.path.realpath(__file__))
    def __init__(self, net_actions=NetActions):
        super().__init__(net_actions=net_actions)

if __name__ == '__main__':
    controller = Controller()
    controller.run()
