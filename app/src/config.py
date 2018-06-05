# -*- coding: utf-8 -*-

import os.path
import configparser


def get_value(section, parameter):
    
	config_parser = configparser.ConfigParser()
	config_path = os.path.dirname(os.path.realpath(__file__)) + '/config.ini'
	config_parser.read(config_path)
	value = None
    
	try:
		value = config_parser.get(section, parameter)
	except Exception:
		raise Exception("Parameter {} missing from section {} in configuration file {}".format(parameter, section, configPath))
	return value
