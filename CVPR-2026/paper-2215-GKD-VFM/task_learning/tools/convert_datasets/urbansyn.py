from PIL import Image
import os
import argparse

def replace_pixel_value(directory, target_value, new_value):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            with Image.open(file_path) as img:
                img = img.convert("L")  # Ensure the image is in grayscale mode
                data = img.getdata()
                new_data = []
                for item in data:
                    if item == target_value:
                        new_data.append(new_value)
                    else:
                        new_data.append(item)
                img.putdata(new_data)
                img.save(file_path)
            print(f"Processed: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace pixel values in images within a directory.")
    parser.add_argument("directory", help="Directory containing images")
    parser.add_argument("--target_value", type=int, default=19, help="Pixel value to be replaced (default: 19)")
    parser.add_argument("--new_value", type=int, default=255, help="New pixel value (default: 255)")

    args = parser.parse_args()
    replace_pixel_value(args.directory, args.target_value, args.new_value)
