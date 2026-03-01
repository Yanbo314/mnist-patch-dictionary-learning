import cv2

def convert_to_ascii(image_path, width=100):
    # 1. Character palette (from darkest to lightest)
    # You can customize this! More characters = more detail.
    CHARS = "@%#*+=-:. "
    
    # 2. Load and preprocess
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found!")
        return
    
    # Convert to grayscale to get brightness values (0-255)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Resize
    # Note: Terminal characters are usually twice as tall as they are wide, 
    # so we multiply the height by 0.5 to prevent the image from looking "stretched".
    h, w = gray_img.shape
    new_h = int(width * (h / w) * 0.5)
    resized_img = cv2.resize(gray_img, (width, new_h))
    
    # 4. The Mapping Magic
    ascii_rows = []
    for row in resized_img:
        line = ""
        for pixel in row:
            # Map 0-255 to 0-9 (index of CHARS)
            index = pixel * (len(CHARS) - 1) // 255
            line += CHARS[index]
        ascii_rows.append(line)
    
    return "\n".join(ascii_rows)

if __name__ == "__main__":
    # Test it with your image
    art = convert_to_ascii("test.jpg")
    if art:
        print(art)
        # Optional: Save to a text file to see the full version
        with open("ascii_result.txt", "w") as f:
            f.write(art)