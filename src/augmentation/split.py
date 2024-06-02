import albumentations as A
import numpy as np
import os
import shutil
from PIL import Image



"""
Returns a list of augmented images.
"""
def augment_image(image):
    new_images = []
    new_images.extend(add_contrast(image))
    new_images.extend(add_blur(image))
    new_images.extend(add_noise(image))
    new_images.extend(add_distortion(image))
    return new_images

"""
Add contrast to the image.
"""
def add_contrast(image):
    rbc = A.RandomBrightnessContrast(p=1.0)
    augmented_rbc = rbc(image=image)
    return [augmented_rbc['image']]

"""
Add blur to the image.
"""
def add_blur(image):
    blur = A.Blur(p=1.0)
    mblur = A.MedianBlur(p=1.0)
    gblur = A.GaussianBlur(p=1.0)
    augmented_blur = blur(image=image)
    augmented_mblur = mblur(image=image)
    augmented_gblur = gblur(image=image)
    return [augmented_blur['image'], augmented_mblur['image'], augmented_gblur['image']]

"""
Add noise to the image.
"""
def add_noise(image):
    gnoise = A.GaussNoise(p=1.0)
    mnoise = A.MultiplicativeNoise(p=1.0)
    augmented_gnoise = gnoise(image=image)
    augmented_mnoise = mnoise(image=image)
    return [augmented_gnoise['image'], augmented_mnoise['image']]

"""
Add distortion to the image.
"""
def add_distortion(image):
    odist = A.OpticalDistortion(p=1.0)
    gdist = A.GridDistortion(p=1.0)
    augmented_odist = odist(image=image)
    augmented_gdist = gdist(image=image)
    return [augmented_odist['image'], augmented_gdist['image']]


"""
Split data into train, train_augment, val, test.
"""
if __name__ == "__main__":
    dataset = np.asarray(os.listdir("./captcha_data/"), dtype=str)

    num_data = dataset.shape[0]
    num_train = int(0.70 * num_data) + 1
    num_val = int(0.15 * num_data)

    shuffle_indices = np.arange(num_data)
    np.random.shuffle(shuffle_indices)

    os.mkdir("./captcha_data/train")
    os.mkdir("./captcha_data/train_augmented")
    os.mkdir("./captcha_data/val")
    os.mkdir("./captcha_data/test")

    for n in list(shuffle_indices[:num_train]):
        shutil.move("./captcha_data/" + dataset[n], "./captcha_data/train/" + dataset[n])
    for n in list(shuffle_indices[num_train:num_train+num_val]):
        shutil.move("./captcha_data/" + dataset[n], "./captcha_data/val/" + dataset[n])
    for n in list(shuffle_indices[num_train+num_val:]):
        shutil.move("./captcha_data/" + dataset[n], "./captcha_data/test/" + dataset[n])

    for f in os.listdir("./captcha_data/train/"):
        label, ext = os.path.splitext(f)
        image = Image.open("./captcha_data/train/" + f)
        image.save("./captcha_data/train_augmented/" + f)
        image = np.array(image)
        augmented_images = augment_image(image)
        for i in range(len(augmented_images)):
            save_path = "./captcha_data/train_augmented/" + label + "-" + str(i) + ext
            augmented_image = Image.fromarray(augmented_images[i])
            augmented_image.save(save_path)
