# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument, protected-access, unused-import, too-many-return-statements
"""transforms on an image which return an image
"""
from random import shuffle as _shuffle
from random import uniform as _uniform

import cv2 as _cv2  # noqa
import numpy as _np

import funclite.baselib as _baselib

from opencvlite import getimg as _getimg  # noqa
import opencvlite.decs as _decs


# region Handling Transforms in Generators
class Transform:
    """ class to hold and execute a transform

    Transforms should all take img as the first argument,
    hence we should be able to also store cv2 or other
    functions directly.

    Where we cant store 3rd party lib transforms directly
    we will wrap them in transforms.py

    func: the function
    *args, **kwargs: func arguments
    p: probability of execution, between 0 and 1.
    """

    def __init__(self, func, *args, p=1, **kwargs):
        """the function and the arguments to be applied

        p is the probability it will be applied
        """
        self._args = args
        self._kwargs = kwargs
        self._func = func
        self.p = p if 0 <= p <= 1 else 1
        self.img_transformed = None

    @_decs.decgetimgmethod
    def exectrans(self, img):
        """(str|ndarray, bool)->ndarray
        Perform the transform on passed image.
        Returns the transformed image and sets
        to class instance variable img_transformed
        """

        if img is not None:
            img_transformed = self._func(img, *self._args, **self._kwargs)
            if isinstance(img_transformed, _np.ndarray):
                self.img_transformed = img_transformed
            elif isinstance(img_transformed, (list, tuple)):
                self.img_transformed = _baselib.item_from_iterable_by_type(img_transformed, _np.ndarray)
            else:
                raise ValueError('Unexpectedly failed to get ndarray image from transforms.exectrans. Check the transformation function "%s" returns an ndarray.' % self._func.__name__)
            return self.img_transformed

        return None


class Transforms:
    """
    Queue transforms and apply to image in FIFO order.

    Args:
        img: ndarray, loaded as ndarray if str
        img_transformed: img after applying the queued transforms

    Methods:
        add: add a transform to the back of the queue
        shuffle: randomly shuffle the transforms
        executeQueue:   apply the transforms to self.img, or
                        pass in a new image


    Examples:
        >>> from opencvlite.transforms import Transforms as t  # noqa
        >>> t1 = t.Transform(t.brightness, p=0.5, value=50)  # noqa
        >>> t2 = t.Transform(t.gamma, gamma=0.7)  # noqa
        >>> t3 = t.Transform(t.rotate, angle=90)  # noqa
        >>> ts = t.Transforms(t1, t2, t3)  # noqa
        >>> ts.shuffle()  # noqa
        >>> ts.executeQueue('C:/temp/myimg.jpg')  # noqa
        >>> cv2.imshow(ts.img_transformed)  # noqa
    """

    def __init__(self, *args, img=None):
        self._img = _getimg(img)
        self.img_transformed = None
        self._tQueue = []
        self._tQueue.extend(args)

    def __call__(self, img=None, execute=True):
        """(str|ndarray) -> void
        Set image if not done previously
        """
        if img is not None:
            self._img = _getimg(img)

        if execute:
            self.executeQueue()

    def add(self, *args):
        """(Transform|Transforms) -> void
        Queue a transform or many transforms.

        Examples:
            >>> t1 = t.Transform(t.brightness, value=50)  # noqa
            >>> ts = t.Transforms(t1) #initialise a transforms instance and queue 1 transform # noqa
            >>> t2 = t.Transform(t.gamma, gamma=0.7) # noqa
            >>> t3 = t.Transform(t.rotate, angle=90)  # noqa
            >>> ts.add(t2, t3) #add 2 more transforms to the queue
        """
        s = 'Queued transforms ' + ' '.join([f._func.__name__ for f in args])  # noqa
        self._tQueue.extend(args)

    def shuffle(self):
        """
        inplace random shuffle of the transform queue
        """
        _shuffle(self._tQueue)

    def executeQueue(self, img=None, print_debug=False):
        """(str|ndarray)->ndarray
        Execute transformation, FIFO order and
        set img_transformed property to the transformed
        image. Also returns the transformed image.

        Args:
            img:
                Image file path or ndarray of image.
            print_debug:
                prints the transforms to console.

        Returns:
            transformed image as ndarray
        """
        if img is not None:
            self._img = _getimg(img)

        first = True
        if _baselib.isempty(self._tQueue):
            return self._img

        for T in self._tQueue:
            pp = _uniform(0, T.p)
            if T.p < pp:
                if print_debug:
                    print('Skipped %s [%s, %s]. (%.2f < %.2f)' % (T._func.__name__, T._args, T._kwargs, T.p, pp))  # noqa
                break

            if print_debug:
                print('Executing %s [%s, %s]' % (T._func.__name__, T._args, T._kwargs))  # noqa

            assert isinstance(T, Transform)
            if first:
                self.img_transformed = T.exectrans(self._img)
                first = False
            else:
                self.img_transformed = T.exectrans(self.img_transformed)

        return self.img_transformed


# endregion


def resize(image, width=None, height=None, inter=_cv2.INTER_AREA, do_not_grow=False):
    """(ndarray|str, int, int, constant, bool)->ndarray
    Resize an image, to width or height, maintaining the aspect.

    image:
        an image or path to an image
    width:
        width of image
    height:
        height of image
    inter:
        interpolation method
    do_not_grow:
        do not increase the image size

    Returns:
        An image

    Notes:
        Returns original image if width and height are None.
        If width or height or provided then the image is resized
        to width or height and the aspect ratio is kept.

        If both width and height are provided, then the image is resized to that width & height
    """
    image = _getimg(image)
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if do_not_grow:
        if height and width:
            if h <= height and w <= width: return image
        if height and not width:
            if h <= height: return image
        if width and w <= width: return image

    if width is not None and height is not None:
        dim = (width, height)
    elif width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    dim = (int(dim[0]), int(dim[1]))
    return _cv2.resize(image, dim, interpolation=inter)


def rotate(image, angle, no_crop=True):
    """(str|ndarray, float, bool) -> ndarray
    Rotate an image through 'angle' degrees.

    image:
        the image as a path or ndarray
    angle:
        angle, positive for anticlockwise, negative for clockwise
    no_crop:
        if true, the image will not be cropped

    """
    img = _getimg(image)
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = _cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    if no_crop:
        cos = _np.abs(M[0, 0])
        sin = _np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return _cv2.warpAffine(img, M, (nW, nH))

    return _cv2.warpAffine(img, M, (w, h))


def perspective_transform(image, pts_orig, pts_trans, **kwargs):
    """
    Do a perspective transform on an image
    Args:
        image: numpy array or file path
        pts_orig: 2d-iterable of 4 points [[0,0], [1,0], ...]
        pts_trans: 2d-iterable of 4 points [[0,0], [1,0], ...] to which pts_orig are transformed in transformed image
        **kwargs: keywords to pass to cv2.warpPerspective

    Notes:
        Points are matched by their list index

    Returns: numpy.ndarray: The image

    Examples
        >>> img_ = perspective_transform('c:/img.jpg', [[0,0], [0,100], [100,100], [100,0]], [[0,0], [10,100], [90,100], [100,0]])

    """
    img = _getimg(image)
    M = _cv2.getPerspectiveTransform(pts_orig, pts_trans)
    out = _cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), **kwargs)
    return out
