import cv2
import numpy as np

# Load the image
image_name = "4k_01_01_22_04-c.jpg"
img = cv2.imread(image_name)

if img is not None:
    # 1. Gentler Denoising
    # Dropped 'h' and 'hColor' from 10 to 5. This removes the worst noise while leaving micro-textures intact.
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    # 2. Unsharp Masking (Sharpening)
    # This isolates the edges and boosts their contrast to bring the sharpness back.
    # Increase the sigma value (2.0) if you need a wider sharpening radius.
    gaussian_blur = cv2.GaussianBlur(denoised_img, (0, 0), 2.0)
    
    # The weights (1.5 and -0.5) control the intensity of the sharpening. 
    # If it looks too harsh, try (1.2, -0.2, 0).
    sharpened_img = cv2.addWeighted(denoised_img, 1.5, gaussian_blur, -0.5, 0)

    # Save the output
    output_name = "enhanced_" + image_name
    cv2.imwrite(output_name, sharpened_img)
    print(f"Image successfully enhanced. Saved as {output_name}")
else:
    print(f"Error: Could not load {image_name}. Ensure it is in the same directory.")