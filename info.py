"""info about images"""
from enum import Enum as _Enum
from glob import glob as _glob
import os.path as _path
import imghdr as _imghdr

import numpy as _np
import cv2 as _cv2
import fuckit as _fuckit

import opencvlite.decs as _decs
from opencvlite import getimg as _getimg  # noqa
from funclite.baselib import list_not as _list_not

__all__ = ['Info', 'ImageInfo']


class eImgType(_Enum):
    """bitwise enum to hold img information"""
    TRUE_BW = 2 ** 0
    COLOR_BW = 2 ** 1
    COLOR_COLOR = 2 ** 2
    COLOR_UNKNOWN = 2 ** 3
    CHANNEL_1 = 2 ** 4
    CHANNEL_2 = 2 ** 5
    CHANNEL_3 = 2 ** 6
    CHANNEL_4 = 2 ** 7
    CHANNEL_UNKOWN = 2 ** 8
    DEPTH8BIT = 2 ** 9
    DEPTH16BIT = 2 ** 10
    DEPTH32BIT = 2 ** 11
    DEPTH_FLOAT = 2 ** 12
    DEPTH_UNKNOWN = 2 ** 13


class Info:
    """general info about versions, system etc"""

    def __init(self):
        pass

    @staticmethod
    def is_cv2():
        """if we are using OpenCV 2, then our _cv2.__version__ will start
        with '2.'
        """
        return Info.opencv_check_version("2.")

    @staticmethod
    def is_cv3():
        """
        if we are using OpenCV 3.X, then our _cv2.__version__ will start
        with '3.'
        """
        return Info.opencv_check_version("3.")

    @staticmethod
    def opencv_check_version(major):
        """(int,_cv2 library)->bool
        Checks major version of _cv2.
        eg. opencv_check_version("3.") will be true for any _cv2 version 3.x.x

        If the supplied library is None, _cv2 will be imported
        """
        return _cv2.__version__.startswith(major)  # noqa


# stupid to make this a class, but stuck with it as loads of refs to it
class ImageInfo:
    """general info about an image"""
    silent = False

    @staticmethod
    @_decs.decgetimgpil
    def is_higher_res(img, w, h, either=False):
        """(ndarray|str, int, int, bool)->bool
        Check if img is
        If either is false, image is considered higher res
        if both w and h are greater than the image w, h
        """
        x, y = ImageInfo.resolution(img)
        if either:
            if w is not None:
                return x > w
            if h is not None:
                return y > h
            return False

        if w is not None and h is not None:
            return x > w and y > h

        return False

    @staticmethod
    @_decs.decgetimg
    def _isbw(img):
        """this is a helper function and shouldnt be used directly.
        It checks if multichannel images are black and white
        """
        # img is a numpy.ndarray, loaded using cv2.imread
        if len(img.shape) > 2:
            looks_like_rgbbw = False not in ((img[:, :, 0:1] == img[:, :, 1:2]) == (img[:, :, 1:2] == img[:, :, 2:3]))
            looks_like_hsvbw = not (True in (img[:, :, 0:1] > 0) or True in (img[:, :, 1:2] > 0))
            return looks_like_rgbbw or looks_like_hsvbw

        return True

    @staticmethod
    def isbw(img, single_channel_only=False):
        """(str, ndarray) -> bool
        Is img black and white, even if it has multichannels
        Returns None if img invalid"""

        img = _getimg(img)
        if not isinstance(img, _np.ndarray):
            return None

        if single_channel_only:
            return bool(eImgType.CHANNEL_1.value & ImageInfo.typeinfo(img))

        return bool(eImgType.COLOR_BW.value & ImageInfo.typeinfo(img))

    @staticmethod
    @_decs.decgetimg
    def sharpval(img):
        """(ndarray|str)->float
        Calculate metric for blurriness of an image
        """
        return _cv2.Laplacian(img, _cv2.CV_64F).var()

    @staticmethod
    def typeinfo(img):
        """(ndarray|str)->int
        Return information about the image for bitwise
        comparison with enum eImgType

        Returns: integer, which can be compared bitwise against Enum:eImageType

        Example:
            >>> ImageInfo(TrueGreyImage).typeinfo & eImageType.CHANNEL_1.value  # noqa
            True
        """
        img = _getimg(img)
        out = 0
        if img.dtype == 'uint8':
            out += eImgType.DEPTH8BIT.value
        elif img.dtype == 'uint16':
            out += eImgType.DEPTH16BIT.value
        elif img.dtype == 'uint32':
            out += eImgType.DEPTH32BIT.value
        elif 'float' in str(img.dtype):
            out += eImgType.DEPTH_FLOAT.value
        else:
            out += eImgType.DEPTH_UNKNOWN.value
            raise ValueError('Unknown image color depth. Numpy array type not IN uint8, uint16, uint32, float.')

        if len(img.shape) == 2:
            out += eImgType.TRUE_BW.value
            out += eImgType.CHANNEL_1.value
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                out += eImgType.CHANNEL_3.value
                if ImageInfo._isbw(img):
                    out += eImgType.COLOR_BW.value
                else:
                    out += eImgType.COLOR_COLOR.value
            elif img.shape[2] == 4:
                out += eImgType.CHANNEL_4.value
                if ImageInfo._isbw(img):
                    out += eImgType.COLOR_BW.value
                else:
                    out += eImgType.COLOR_COLOR.value
            elif img.shape[2] == 2:
                out += eImgType.CHANNEL_2.value
                out += eImgType.COLOR_UNKNOWN.value
        return out

    @staticmethod
    @_decs.decgetimgpil
    def is_lower_res(img, w, h, either=True):
        """(ndarray|str, int, int, bool)->bool
        Check if img is lower res than that defined by w and h.
        If either is true, image is considered lower res
        if either the width or the height is less than w, h
        """
        x, y = ImageInfo.resolution(img)
        if either:
            if w is not None:
                return x < w
            if h is not None:
                return y < h
            return False

        if w is not None and h is not None:
            return x < w and y < h

        return False

    @staticmethod
    @_decs.decgetimgpil
    def resolution(img):
        """(str|ndarray)->int,int
        width,height
        """
        if isinstance(img, _np.ndarray):
            h, w = img.shape[:2]
            return w, h

        return [img.width, img.height]

    @staticmethod
    def is_image(file_path, try_load=False):
        """(str, bool) ->b ool
        Pass in a file string and see if it looks like an image.

        try_load:
            Try and loading file_path with _cv2.imread (costly)

        Example:
            >>> is_image('C:/temp/picture.jpg', try_load=False)  # noqa
            True
        """
        ret = False
        file_path = _path.normpath(file_path)

        if not _path.isfile(file_path):
            return False

        if not _imghdr.what(file_path) is None:
            ret = True
            if try_load:
                try:
                    ret = False
                    img = _cv2.imread(file_path)
                    if isinstance(img, _np.ndarray):
                        ret = True
                except Exception as _:
                    pass

        return ret

    @staticmethod
    def get_image_resolutions(glob_str, unique=True, as_row_col=False):
        """(str, bool, bool)-> depth-2 list
        Returns all image resolutions (COLUMN, ROW) as
        list of lists

        glob_str:
            Wildcarded file system path
        unique:
            return unique resolutions only, otherwise
            every resolution will be returned
        as_row_col:
            returns as row, col

        Examples:
            >>> get_image_resolutions('C:/temp/*.jpg', False)  # noqa
            [[800,600], [1024,768]]

            >>> get_image_resolutions('C:/temp/*.jpg', True)  # noqa
            [[800,600], [1024,768], [800,600]]
        """
        dims = []
        for pic in _glob(glob_str):
            if ImageInfo.is_image(pic):  # does the work
                with _fuckit:
                    img = _cv2.imread(pic)

                if isinstance(img, _np.ndarray):
                    h, w = img.shape[:2]
                    if as_row_col:
                        e = [h, w]
                    else:
                        e = [w, h]

                    if e not in dims or not unique:
                        dims.append(e)
        return dims

    @staticmethod
    def getsize(img):
        """(ndarray|str)->int, int
        return width and height of an image (in that order)

        Can pass in a file path or ndarray
        """
        if isinstance(img, str):
            img = _cv2.imread(img)

        h, w = img.shape[:2]
        return w, h

    @staticmethod
    def getsize_dict(img):
        """(ndarray|str)->{'w':cols,'h':rows}
        return width and height of an image (in that order)

        Can pass in a file path or ndarray
        """
        if isinstance(img, str):
            img = _cv2.imread(img)

        h, w = img.shape[:2]
        return {'w': w, 'h': h}


def check(img, enum_types):
    """(ndarray|str, Enum:eImageType|List:Enum:eImageType)->bool
    Quick check if image is enum_type, or enum_types can be
    a list of eImageTypes, in which case check for all
    conditions

    Parameters:
        img (ndarray|str): ndarray or path to image
        enum_types: A single eImageType enumeration, or a list of eImageTypes

    Returns: True if image meets enum_type

    Example:
        >>> check(TrueGreyImage, eImageType.CHANNEL_1)  # noqa
        True
    """
    img = _getimg(img)
    image_enums = []
    if isinstance(enum_types, eImgType):
        enum_types = [enum_types]

    if img.dtype == 'uint8':
        image_enums.append(eImgType.DEPTH8BIT)
    elif img.dtype == 'uint16':
        image_enums.append(eImgType.DEPTH16BIT)
    elif img.dtype == 'uint32':
        image_enums.append(eImgType.DEPTH32BIT)
    elif 'float' in str(img.dtype):
        image_enums.append(eImgType.DEPTH_FLOAT)
    else:
        image_enums.append(eImgType.DEPTH_UNKNOWN)
        raise ValueError('Unknown image color depth. Numpy array type not IN uint8, uint16, uint32, float.')

    if len(img.shape) == 2:
        image_enums.append(eImgType.TRUE_BW)
        image_enums.append(eImgType.CHANNEL_1)
    elif len(img.shape) == 3:
        if img.shape[2] == 3:
            image_enums.append(eImgType.CHANNEL_3)
            if ImageInfo._isbw(img):  # noqa
                image_enums.append(eImgType.COLOR_BW)
            else:
                image_enums.append(eImgType.COLOR_COLOR)
        elif img.shape[2] == 4:
            image_enums.append(eImgType.CHANNEL_4)
            if ImageInfo._isbw(img):  # noqa
                image_enums.append(eImgType.COLOR_BW)
            else:
                image_enums.append(eImgType.COLOR_COLOR)
        elif img.shape[2] == 2:
            image_enums.append(eImgType.CHANNEL_2)
            image_enums.append(eImgType.COLOR_UNKNOWN)

    return not bool(_list_not(enum_types, image_enums))


def isbright(image, dim=20, thresh=0.5):
    """(ndarray, int, float) -> bool
    Check if an image is bright
    image: ndarray
    dim: resize dimensions to reduce computation
    thresh: 0-1, threshhold to test if image is bright.

    Returns: bool,  1 is bright. 0 is not
    """

    # Resize image to 10x10
    image = _cv2.resize(image, (dim, dim))

    # Convert color space to LAB format and extract L channel
    L, A, B = _cv2.split(_cv2.cvtColor(image, _cv2.COLOR_BGR2LAB))

    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L / _np.max(L)
    # Return True if mean is greater than thresh else False
    return _np.mean(L) > thresh


def isblurry(image, thresh=100):
    """(ndarray, float) -> bool
    Detect if an image is blurry.
    image: ndarray
    thresh: float
    Returns: bool, true if the laplacian is less than thresh.
    """
    v = _cv2.Laplacian(image, _cv2.CV_64F).var()
    return v < thresh
