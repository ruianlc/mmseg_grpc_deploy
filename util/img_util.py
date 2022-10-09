"""
Author : Arnold Yeung, 2020.11.07

Main functions for image rotation service.
Modified by An., 2022.09.20. 
"""

from PIL import Image
import base64
from io import BytesIO


#   global variables
MAX_WIDTH = 1000 
MAX_HEIGHT = 660

def convert_to_jpg(file):
    """
    Convert a files (e.g., .png) to .jpg.
    Arguments:
        - file (str) :          file to convert to .jpg
    Returns:
        - file (str) :          file of the converted .jpg
    """

    extension = file[-4:]
    if extension not in ['.jpg', 'jpeg']:
        print("Converting file to jpg")
        img = Image.open(file)
        new_file = file[:-3]+"jpg"
        img.save(new_file)
        return new_file
    else:
        return file

def resize_image_with_aspect_ratio(image, size=(1000, 1000)):
    """
    Resize a PIL.Image while maintaining aspect ratio.
    Arguments:
        - image (PIL.Image) :           image to resize
        - size (tuple(int, int)) :      maximum width and height by pixels
    Returns:
        - image (PIL.Image) :           resized image 
    """
    image.thumbnail(size, Image.ANTIALIAS)
    return image

def convert_bytes_to_image(byte_data, width, height):
    """
    Transforms byte data of an image to PIL.Image object.
    Arguments:
        - byte_data (bytes) :           byte string containing image data row-wise (e.g., pixel values)
        - width (int) :                 number of pixels wide
        - height (int) :                number of pixels tall
        - rgb (bool) :                  whether the image is colored (or greyscaled)
    Returns:
        - img (PIL.Image) :             PIL.Image object for the given byte data
    """        
    img = Image.frombytes(mode="RGB", size=(width, height), data=byte_data)
    return img

def rotate_image(request):
    """
    Rotates an image by the designated degree.
    Arguments:
        - request (ABImageRotateRequest) :
            - rotation (int) :              number of 90 degrees to rotate ccw
            - image (ABImage) :     ABImage object containing data (bytes) and other attributes
                - data (byte) :     byte data of image (e.g., pixel values)
                - width (int) :     number of pixels wide
                - height (int) :    number of pixels tall
                - color (bool) :    whether image is colored (or greyscale)
    Returns:
        - img (PIL.Image) :         rotated image object
    """

    rotate_degree = request.rotation * 90

    img = convert_bytes_to_image(request.image.data, 
                                request.image.width, 
                                request.image.height, 
                                request.image.color)
    img = img.rotate(angle=rotate_degree, expand=True)

    return img

def convert_image_to_ABImage(image, rgb=True):
    """
    Converts a PIL.Image object to ABImage.
    Arguments:
        - image (PIL.Image) :               Image object to convert
        - rgb (bool) :                      whether image is colored (or greyscaled)
    Returns:
        - out (ABImage) :                   converted ABImage object 
    """

    img_data = []

    #   resize image
    size = (MAX_WIDTH, MAX_HEIGHT)

    if image.width > MAX_WIDTH or image.height > MAX_HEIGHT:
        print("Image is too large. Resizing...")
        image = resize_image_with_aspect_ratio(image, size)

    if rgb:
        img_rgb = list(image.getdata())

        for rgb_pixel in img_rgb:
            try:
                for rgb_point in rgb_pixel:
                    img_data.append(rgb_point)
            except:
                raise ValueError("Color argument may not correspond to input image.")
    else:
        img_greyscale = image.convert("L")
        
        for row in range(0, image.height):
            for col in range(0, image.width):
                img_data.append(img_greyscale.getpixel((col, row)))
    out = {
        "width": image.width,
        "height": image.height,
        "data": bytes(img_data),
    }

    return out

def convert_imagefile_to_ABImage(infile, rgb=True):
    """
    Converts an image_file (e.g., .jpg) to an ABImage object.
    Arguments:
        - infile (str) :            image file to convert
        - rgb (bool) :              whether image is colored or not
    Returns:
        - out (ABImage) :           equivalent ABImage object
    """

    img = Image.open(infile, mode='r')
    out = convert_image_to_ABImage(img, rgb)
    return out

def save_image(image, outfile):
    """
    Save an PIL.Image object into a file.
    Arguments:
        - image (PIL.Image) :       image object to save
        - outfile (str) :           file to save the image to 
    Returns:
        None
    """
    image.save(outfile)
    
def convert_PIL_to_base64(img_PIL):
    """
    Date: 2022.10.09. borrowed from https://jdhao.github.io/
    PIL.Image object to base64 string.
    Arguments:
        - img_PIL (PIL.Image) :     image object to convert
    Returns:
        - img_base64 (string) :     base64 image string
    """
    im_file = BytesIO()
    img_PIL.save(im_file, format="png")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    img_base64 = base64.b64encode(im_bytes)

    return img_base64

def convert_base64_to_PIL(img_base64):
    """
    Date: 2022.10.09. borrowed from https://jdhao.github.io/
    base64 image string to PIL.Image object.
    Arguments:
        - img_base64 (string) :     base64 image string to convert
        
    Returns:
        - img_PIL (PIL.Image) :     PIL.Image object
    """
    im_bytes = base64.b64decode(img_base64)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    img_PIL = Image.open(im_file)   # img is now PIL Image object

    return img_PIL