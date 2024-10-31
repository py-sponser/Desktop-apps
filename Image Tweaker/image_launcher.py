from image_tweaker import ImageTweaker
import cv2, eel, wx, os, shutil
from double_image_launcher import get_second_image
import matplotlib.pyplot as plt


eel.init("web")  # initializing eel to display web files that exist in the 'web' directory
tweaker = ImageTweaker()  # defining an object of ImageTweaker() class
file_name = str()  # will be treated as global variable to have the chosen image name saved.
file_original_path = str()  # will be treated as global variable to have the image original path saved.
to_redo_list = list()  # list for popped images of tweaker.image_states when undo
blur_filters = ["gauss", "average", "circular", "pyramidal", "cone", "median"]


@eel.expose  # exposing the below function to javascript (so we'll be able to call this python function in javascript code)
def switch_to_grayed():
    image = tweaker.switch_to_grayed_scale()
    cv2.imwrite(tweaker.image_path, image)  # overriding image each time an operation is done on it.


@eel.expose
def show_image_data():
    image_data = tweaker.get_image_data()
    eel.getImageData(image_data)  # executing a javascript function that were exposed to python.


@eel.expose
def translate_image(tx, ty):
    if tx or ty:
        image = tweaker.translate_image(tx=float(tx), ty=float(ty))
        cv2.imwrite(tweaker.image_path, image)


@eel.expose
def rotate_image(angle):
    if angle:
        image = tweaker.rotate_image(angle=angle)
        cv2.imwrite(tweaker.image_path, image)


@eel.expose
def skew_image(skew_value):
    if skew_value:
        image = tweaker.skew_image(skew_value)
        cv2.imwrite(tweaker.image_path, image)


@eel.expose
def deskew_image(deskew_value):
    if deskew_value:
        image = tweaker.deskew_image(deskew_value)
        cv2.imwrite(tweaker.image_path, image)


@eel.expose
def zoom_image(zoom_area):
    if zoom_area:
        tweaker.zoom_image(zoom_area)


@eel.expose
def flip_image(axis_code):
    if axis_code == "x":
        code = 1
    elif axis_code == "y":
        code = 0
    else:
        code = -1

    image = tweaker.flip_image(axis_code=code)
    cv2.imwrite(tweaker.image_path, image)


@eel.expose
def get_image(wildcard="*"):
    """Displaying Dialog for user to select image"""
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)  # configuring how dialog displays and looks
    if dialog.ShowModal() == wx.ID_OK:  #
        global file_original_path  # considering file_original_path variable here as global
        file_original_path = dialog.GetPath()  # saving the selected file path
        global file_name  # considering file_original variable here as global
        file_name = dialog.GetFilename()  # saving the name of the selected file
        shutil.copyfile(file_original_path, f"{os.path.dirname(os.path.realpath(__file__))}/web/images/{file_name}")
        # copying the file from its path, to project_directory_path/web/images/ directory
        image_path = f"web/images/{file_name}"  # saving image_path especially to this current python file 'image_launcher'
        tweaker.set_image_conf(image_path)  # setting ImageTweaker configurations
        return file_name # returning file_name to javascript as to be the path to look for the images using html img tag
    else:
        path = None
    dialog.Destroy()  # finally, close the dialog


@eel.expose
def blind_image():
    second_image_path = get_second_image()
    read_second_image = cv2.imread(second_image_path, 0)
    blinded_image = tweaker.blind_image_points(read_second_image)
    cv2.imwrite(tweaker.image_path, blinded_image)


@eel.expose
def display_histogram():
    tweaker.histogram_display()


@eel.expose
def equalize_image():
    equalized_image = tweaker.histogram_equalize()
    cv2.imwrite(tweaker.image_path, equalized_image)


@eel.expose
def correct_gama(gamma_power_value):
    if gamma_power_value:
        corrected_gamma_image = tweaker.correct_gama_power(gamma_power_value)
        cv2.imwrite(tweaker.image_path, corrected_gamma_image)


@eel.expose
def transform_linear():
    linear_transfromed_image = tweaker.transform_linear()
    cv2.imwrite(tweaker.image_path, linear_transfromed_image)


@eel.expose
def log_transform():
    log_transformed_image = tweaker.log_transform()
    cv2.imwrite(tweaker.image_path, log_transformed_image)


@eel.expose
def slice_gray_level():
    clarified_image = tweaker.slice_gray_level()
    cv2.imwrite(tweaker.image_path, clarified_image)


@eel.expose
def compress():
    compressed_image = tweaker.compress()
    cv2.imwrite(tweaker.image_path, compressed_image)


@eel.expose
def show_compress_planes():
    image_of_planes = tweaker.show_compress_planes()
    cv2.imwrite(tweaker.image_path, image_of_planes)


@eel.expose
def filter_image(filter_type, pixel_range_dst=None):
    filter_type = str(filter_type).lower()
    print(filter_type)
    print(pixel_range_dst)
    if filter_type in blur_filters:
        # filtered_image = filters[filter_type](pixel_range_dst)
        filtered_total_image = tweaker.filter(filter_type, pixel_range_dst)
        try:
            if filtered_total_image.any():
                cv2.imwrite(tweaker.image_path, filtered_total_image)
        except:
            pass
# Test filtering ranged image.


def show_filter_blurred_image():
    eel.showFilterBlurredImage()


@eel.expose
def median_blur():
    median_blurred_image = tweaker.median_blur()
    cv2.imwrite(tweaker.image_path, median_blurred_image)


@eel.expose
def filter_sobel(sobel_filter_type):
    if sobel_filter_type:
        sobel_filtered_image = tweaker.sobel_filter(sobel_filter_type)
        cv2.imwrite(tweaker.image_path, sobel_filtered_image)


@eel.expose
def filter_laplace():
    laplace_filtered_image = tweaker.laplace()
    cv2.imwrite(tweaker.image_path, laplace_filtered_image)


@eel.expose
def segment():
    segmented_image = tweaker.segment_image()
    cv2.imwrite(tweaker.image_path, segmented_image)


@eel.expose
def undo_image():
    image_states_length = len(tweaker.image_states) - 1  # saving the actual length (in index domain) of image states
    if image_states_length >= 0:   # if length is bigger than 0
        to_redo_list.append(tweaker.image_states.pop())  # pop the last element of image states and append it to to_redo_list
        tweaker.image = tweaker.image_states[-1]  # get the last image state in image_states
        cv2.imwrite(tweaker.image_path, tweaker.image)  # override image


@eel.expose
def redo_image():
    image_states_length = len(tweaker.image_states) - 1  # saving the actual length (in index domain) of image states
    tweaker.image_states.append(to_redo_list.pop())  # pop the first element of to_redo_list and append it to image_states again
    tweaker.image = tweaker.image_states[-1]  # get the last image state in image_states which is `to_redo_list.pop(0)`
    cv2.imwrite(tweaker.image_path, tweaker.image) # override image


@eel.expose
def reset_image():
    """Resetting the image just by re-copying the original image to the project folder path again"""
    global file_name  # getting image name
    global file_original_path  # getting the original image path that I saved from the beginning
    image_path = f"web/images/{file_name}"  # setting image path in project images directory
    shutil.copyfile(file_original_path, f"{os.path.dirname(os.path.realpath(__file__))}/web/images/{file_name}")
    # copying the original image from its original path to the project folder again.
    tweaker.set_image_conf(image_path)  # setting image path in order to get read by the GUI


eel.start("index.html", size=(1280, 1024), position=(300, 300))
