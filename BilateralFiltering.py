import cv2

# Load the image
image_name = "4k_01_01_22_04-c.jpg"
img = cv2.imread(image_name)

if img is not None:
    # 1. Bilateral Filter
    # Parameters: src, d (diameter of pixel neighborhood), sigmaColor, sigmaSpace
    # Higher sigmaColor means more colors will be mixed together (smoothing).
    # Higher sigmaSpace means pixels further away will influence each other.
    denoised_img = cv2.bilateralFilter(img, 9, 150, 150)

    # 2. Mild Unsharp Masking (Sharpening)
    gaussian_blur = cv2.GaussianBlur(denoised_img, (0, 0), 2.0)
    final_img = cv2.addWeighted(denoised_img, 1.5, gaussian_blur, -0.5, 0)

    # Save the output
    output_name = "bilateral_" + image_name
    cv2.imwrite(output_name, final_img)
    print(f"Image successfully enhanced. Saved as {output_name}")
else:
    print(f"Error: Could not load {image_name}. Ensure it is in the same directory.")