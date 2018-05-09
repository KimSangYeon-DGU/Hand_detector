import argparse
import os
import cv2
import numpy as np

def main(args):
    input_dir = args['input_dir']
    output_dir = args['output_dir']
    
    image_names = os.listdir(input_dir)

    for image_name in image_names:
        # Read image
        image = cv2.imread('{0}/{1}'.format(input_dir, image_name), 1)

        # Convert RGB to YCrCb color space
        ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Set lower and upper bound for optimization on skin
        lower_bound = np.array([0, 133, 77], np.uint8)
        upper_bound = np.array([255, 173, 127], np.uint8)

        # Generate skin mask(background: 0, skin: 255)
        mask = cv2.inRange(ycc, lower_bound, upper_bound)

        # Erode mask for sharpness
        mask = cv2.erode(mask, np.ones((1,1), np.uint8), iterations=1)

        # Dilate mask for smoothing
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)

        # Find contours
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours
        cv2.drawContours(image, contours, -1, (0, 0, 0), 3)

        # Generate board to attach parts of skin
        panel = np.zeros_like(image)
        
        panel[mask == 255] = image[mask == 255]

        cv2.imwrite('{0}/{1}'.format(output_dir, image_name), panel);
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='The input images directory path')
    parser.add_argument('--output_dir', required=True, help='The output images directory path')
    
    args = vars(parser.parse_args())
    main(args)
    
