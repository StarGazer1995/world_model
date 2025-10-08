import cv2
import glob
import numpy as np

def main():
    image_folder = "data/nuscenes/samples/CAM_FRONT_LEFT_BK"
    image_files = glob.glob(image_folder + "/*jpg")
    sorted(image_files)
    
    image_sample_path = image_files[0]
    image_sample = cv2.imread(image_sample_path)
    image_shape = image_sample.shape
    fake_image = np.zeros(image_shape, dtype=image_sample.dtype)
    print(image_shape)

    fake_image_paths = [image_path.replace("CAM_FRONT_LEFT_BK", "CAM_FRONT_LEFT") for image_path in image_files]
    for fake_image_path in fake_image_paths:
        cv2.imwrite(fake_image_path, fake_image)
    print("ALL Done")

if __name__ == "__main__":
    main()