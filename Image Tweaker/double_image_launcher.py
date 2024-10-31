import image_tweaker
import cv2, eel, wx, os, shutil

images_count = 0

tweaker = image_tweaker.ImageTweaker()  # defining an object of ImageTweaker() class
file_name = str()  # will be treated as global variable to have the chosen image name saved.
file_original_path = str()  # will be treated as global variable to have the image original path saved.
to_redo_list = list()  # list for popped images of tweaker.image_states when undo


def get_second_image(wildcard="*"):
    """Displaying Dialog for user to select image"""
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)  # configuring how dialog displays and looks
    if dialog.ShowModal() == wx.ID_OK:  # displaying the dialog when user clicks
        global file_original_path  # considering file_original_path variable here as global
        file_original_path = dialog.GetPath()  # saving the selected file path
        global file_name  # considering file_original variable here as global
        file_name = dialog.GetFilename()  # saving the name of the selected file
        shutil.copyfile(file_original_path, f"{os.path.dirname(os.path.realpath(__file__))}/web/images/{file_name}")
        # copying the file from its path, to project_directory_path/web/images/directory
        image_path = f"web/images/{file_name}"  # saving image_path especially to this current python file 'image_launcher'
        tweaker.set_image_conf(image_path)  # setting ImageTweaker configurations
        return image_path # returning file_name to javascript as to be the path to look for the images using html img tag
    else:
        return None
    dialog.Destroy()  # finally, close the dialog

