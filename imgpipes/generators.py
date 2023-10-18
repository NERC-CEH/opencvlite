# pylint: disable=C0103, locally-disabled, attribute-defined-outside-init, protected-access, unused-import, arguments-differ, unused-argument
# TOO Suppress excepions globally for this module when in SILENT mode, check iolib which has the routine to do it
"""Image generatrs for multiple sources.

All yielded generators have an error handler wrapping which logs errors
to prevent stop failures during big processing tasks

NEW GENERATORS
    Filter (sieve) and Transform Support
        When providing new generators, ensure to add delegate filtering
        and transformation by adding the following after the image is obtained:

        img = _cv2.imread(fname, outflag)
        img = super().generate(img) #transforms and checks the filter
    Yield
        All generators should yield ndarray, filepath, dict
        Where dict is generator specific information.
    """

from os import path as _path
import abc as _abc

import logging as _logging
from inspect import getsourcefile as _getsourcefile

import cv2 as _cv2
import numpy
import numpy as _np

import funclite.iolib as _iolib

import opencvlite.decs as _decs

import opencvlite.imgpipes.filters as _filters

import opencvlite.transforms as _transforms
from opencvlite import IMAGE_EXTENSIONS_AS_WILDCARDS as _IMAGE_EXTENSIONS_AS_WILDCARDS

SILENT = True

_pth = _iolib.get_file_parts2(_path.abspath(_getsourcefile(lambda: 0)))[0]
_LOGPATH = _path.normpath(_path.join(_pth, 'features.py.log'))
_logging.basicConfig(format='%(asctime)s %(message)s', filename=_LOGPATH, filemode='w', level=_logging.DEBUG)


def _prints(s, log=True):
    """silent print"""
    if not SILENT:
        print(s)
    if log:
        _logging.propagate = False
        _logging.info(s)
        _logging.propagate = True


_prints('Logging to %s' % _LOGPATH)


# region Generator related classes
class _BaseGenerator(_abc.ABC):
    """abstract class for all these generator functions
    """

    @_abc.abstractmethod
    def generate(self):
        """placeholder"""
        pass


# TODO Implement regex sieve on image filenames
# Do it in the base class so it is used by all
# classes which inherit from _Generator
class _Generator(_BaseGenerator):
    """base generator class
    Use with BaseGenerator to create new generators

    Pops transforms and filters.

    When instantiating classes which inherit Generator
    provide kwargs with
    transforms=Transforms (Transforms class - a collection of Transfor classes)
    filters=Filters (Filters class - a collection of Filter classes)
    """

    def __init__(self, *args, **kwargs):
        self._transforms = kwargs.pop('transforms', None)
        if not isinstance(self._transforms, _transforms.Transforms) and self._transforms is not None:
            raise ValueError('Base generator class keyword argument "transforms" requires class type "transforms.Transforms"')

        self._filters = kwargs.pop('filters', None)
        if not isinstance(self._filters, _filters.Filters) and self._filters is not None:
            raise ValueError('Base generator class keyword argument "filters" requires class type "filters.Filters"')

    # assert isinstance(self._transforms, _transforms.Transforms)
    # assert isinstance(self._filters, _filters.Filters)

    @property
    def transforms(self):
        """transforms getter"""
        return self._transforms

    @transforms.setter
    def transforms(self, transforms):
        """transforms setter"""
        self._transforms = transforms

    @property
    def filters(self):
        """filters getter"""
        return self._filters

    @filters.setter
    def filters(self, filters):
        """filters setter"""
        self._filters = filters

    @_decs.decgetimgmethod
    def executeTransforms(self, img):
        """(ndarray|str)->ndarray
        execute transforms enqueued in the Transforms class
        and return the transformed image
        """
        # img = _getimg(img)
        if isinstance(self._transforms, _transforms.Transforms):
            try:
                img = self.transforms.executeQueue(img)
            except ValueError as e:
                _logging.exception(str(e))
        return img

    @_decs.decgetimgmethod
    def isimagevalid(self, img):
        """does the image pass a filter
        """
        # img = _getimg(img)
        # assert isinstance(self.filters, _filters.Filters)
        if isinstance(self.filters, _filters.Filters):
            return self.filters.validate(img)

        return True

    @_decs.decgetimgmethod
    def generate(self, img):
        """(str|ndarray,cv2.imread flag..)->None|ndarray
        takes img, applies relvant filters and transforms
        and returns to calling generater to bubble up the
        transformed image

        Returns None if filter fails
        """
        if self.isimagevalid(img):
            img = self.executeTransforms(img)
            return img

        return None


# endregion


class FromPaths(_Generator):
    """Generate images from a list of folders
    Transforms and filters can be added by instantiating lists Transform and Filter
    objects and passing them as named arguments. See test_generators for more
    examples.

    paths:
        Single path or list/tuple of paths
    wildcards:
        Single file extension or list of file extensions.
        Extensions should be dotted, an asterix is appended
        if none exists.

    Yields: ndarray, str, dict   i.e. the image, image path, {}

    Example:
        fp = generators.FromPaths('C:/temp', wildcards='*.jpg',
                            transforms=Transforms, filters=Filters)
    """

    def __init__(self, paths, *args, wildcards=_IMAGE_EXTENSIONS_AS_WILDCARDS, **kwargs):
        self._paths = paths
        self._wildcards = wildcards
        super().__init__(*args, **kwargs)

    @property
    def paths(self):
        """paths getter"""
        return self._paths

    @paths.setter
    def paths(self, paths):
        """paths setter"""
        self._paths = paths

    def generate(self, outflag=_cv2.IMREAD_UNCHANGED, pathonly=False, recurse=False):  # noqa
        """(cv2.imread option, bool, bool) -> ndarray, str, dict
        Globs through every file in paths matching wildcards returning
        the image as an ndarray

        recurse:
            Recurse through paths
        outflag:
            <0 - Loads as is, with alpha channel if present)
            0 - Force grayscale
            >0 - 3 channel color iage (stripping alpha if present
        pathonly:
            only generate image paths, the ndarray will be None

        Yields:
            image, path, an empty dictionary

        Notes:
            The empty dictionary is yielded so it is the same format as other generators
         """

        for imgpath in _iolib.file_list_generator1(self._paths, self._wildcards, recurse=recurse):
            try:
                if pathonly:
                    yield None, imgpath, {}
                else:
                    img = _cv2.imread(imgpath, outflag)  # noqa
                    img = super().generate(img)  # delegate to base method to transform and filter (if specified)
                    if isinstance(img, _np.ndarray):
                        yield img, imgpath, {}
            except Exception as _:
                s = 'Processing of %s failed.' % imgpath
                _logging.exception(s)


class FromList(_Generator):
    """Generate images from a list of folders
    Transforms and filters can be added by instantiating lists Transform and Filter
    objects and passing them as named arguments. See test_generators for more
    examples.

    paths:
        Single path or list/tuple of paths
    wildcards:
        Single file extension or list of file extensions.
        Extensions should be dotted, an asterix is appended
        if none exists.

    Yields: ndarray, str, dict   i.e. the image, image path, {}

    Example:
        fp = generators.FromPaths('C:/temp', wildcards='*.jpg',
                            transforms=Transforms, filters=Filters)
    """

    def __init__(self, file_list, *args, **kwargs):
        self._file_list = file_list
        super().__init__(*args, **kwargs)

    @property
    def file_list(self):
        """paths getter"""
        return self._file_list

    @file_list.setter
    def file_list(self, file_list):
        """paths setter"""
        self._file_list = file_list

    def generate(self, outflag: int = _cv2.IMREAD_UNCHANGED, pathonly: bool = False) -> tuple[(None, numpy.ndarray), str, dict]:  # noqa
        """
        Generates every file in the set list or tuple returning
        the image as an ndarray, or optionally, just the file path.

        Args:
            outflag (int): Flag passed to cv2.imread, specifiying properties of the returned image
                <0 - Loads as is, with alpha channel if present)
                0 - Force grayscale
                >0 - 3 channel color iage (stripping alpha if present
            pathonly (bool): Only generate image paths, the ndarray will be None

        Yields:
            tuple[(None,numpy.ndarray), str, dict]: image (ndarray), path, an empty dictionary if pathonly false, otherwise image is None

        Notes:
            The empty dictionary is yielded to match the format as other generators.
            Yielded paths are normpathed.
         """

        for imgpath in self.file_list:
            imgpath = _path.normpath(imgpath)
            try:
                if pathonly:
                    yield None, imgpath, {}
                else:
                    img = _cv2.imread(imgpath, outflag)  # noqa
                    img = super().generate(img)  # delegate to base method to transform and filter (if specified)
                    if isinstance(img, _np.ndarray):
                        yield img, imgpath, {}
            except Exception as _:
                s = 'Processing of %s failed.' % imgpath
                _logging.exception(s)
