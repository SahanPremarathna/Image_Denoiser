import cv2

# Load the image
image_name = "4k_01_01_22_04-c.jpg"
img = cv2.imread(image_name)

if img is not None:
    # Apply Non-Local Means Denoising
    # Parameters: src, dst, h (luminance filter strength), hColor, templateWindowSize, searchWindowSize
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Save the output
    output_name = "denoised_" + image_name
    cv2.imwrite(output_name, denoised_img)
    print(f"Noise successfully removed. Saved as {output_name}")
else:
    print(f"Error: Could not load {image_name}. Ensure it is in the same directory.")