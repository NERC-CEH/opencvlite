"""main init file for package opencvlite

A very restricted set of opencv wrappers, cut down to use in this erammp-python package.

Given the small size and restricted use cases of opencvlite within this erammp-python package,
migrating it to a stand-alone package not justified.
"""
import os.path as _path

import cv2 as _cv2

from opencvlite import *

IMAGE_EXTENSIONS = ('.bmp',
                    '.jpg',
                    '.jpeg',
                    '.png',
                    '.tif',
                    '.tiff',
                    '.pbm',
                    '.pgm',
                    '.ppm')

IMAGE_EXTENSIONS_AS_WILDCARDS = ('*.bmp',
                                 '*.jpg',
                                 '*.jpeg',
                                 '*.png',
                                 '*.tif',
                                 '*.tiff',
                                 '*.pbm',
                                 '*.pgm',
                                 '*.ppm')

__all__ = ['common', 'color', 'decs', 'info', 'transforms', 'IMAGE_EXTENSIONS_AS_WILDCARDS', 'IMAGE_EXTENSIONS']


# Global logger
# To use:
# <package>.info("informational message")
# <package>.debug("debug message")
# <package>.critical("informational message")
# try:
#    import funclite.log as _log
#    _logfile = _path.join(_log.RootLogger.USER_TEMP_FOLDER, 'opencvlite.log')
#    _rootlogger = _log.RootLogger(_logfile)
#    Log = _rootlogger.logger
#    print('Logging to', _logfile)
# except Exception as e:
#    print('Logger initialisation failed for file %s.\nError: %s' % (_logfile, str(e)))


# def loginfo():
#    """print log file status"""
#    try:
#        print(_rootlogger)
#    except Exception as e:
#        print('Failed to get log info for file %s.\nError: %s' % (_logfile, str(e)))


def getimg(img, outflag=_cv2.IMREAD_UNCHANGED):
    """(ndarray|str)->ndarray
    tries to load the image if its a path and returns the loaded ndarray
    otherwise returns input img if it is an ndarray

    Also consider using @decs._decs.decgetimg decorator
    """
    if isinstance(img, str):
        return _cv2.imread(_path.normpath(img), outflag)
    return img
