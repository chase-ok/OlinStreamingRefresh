#!/usr/bin/env 

import tables as tb
import numpy as np


def openStorage(hdf5File='data.h5'):
	return tb.openFile(hdf5File, 'a')

def tableExists(root, tableName):
	return hasattr(root, tableName)

def clearTable(root, tableName):
	if tableExists(root, tableName):
		getattr(root, tableName)._f_remove(recursive=True)