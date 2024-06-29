from PIL import Image
import piexif

def check_image_compression(image_path):
    try:
        with Image.open(image_path) as img:
            format = img.format
            # Common compressed formats
            if format in ["JPEG", "JPG", "PNG", "GIF", "WEBP"]:
                print(f"The image format is {format}, which is typically compressed.")
            else:
                print(f"The image format is {format}, which is less likely to be compressed.")
            
            if format in ["JPEG", "JPG"]:
                print(img.info)
                exif_dict = piexif.load(img.info.get('exif', ''))
                if "0th" in exif_dict and piexif.ImageIFD.Compression in exif_dict["0th"]:
                    compression = exif_dict["0th"][piexif.ImageIFD.Compression]
                    if compression != 1:  # 1 means uncompressed
                        print(f"The JPEG image is compressed. Compression value: {compression}")
                    else:
                        print("The JPEG image is not compressed.")
                else:
                    print("No EXIF compression metadata found for JPEG image.")
            elif format == "PNG":
                if img.info.get("interlace"):
                    print("The PNG image is interlaced (which can be a form of compression).")
                else:
                    print("The PNG image is not interlaced.")
            elif format == "WEBP":
                if img.info.get("lossless", False):
                    print("The WebP image is lossless (not compressed).")
                else:
                    print("The WebP image is compressed.")
            else:
                print(f"No additional compression information available for format: {format}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
image_path = '/home/ubuntu/y1/DistilDIRE/datasets/truemedia-political/images/fakes/ZPNOX270fe2dJ6fHvG1mVaHwHM0.jpg'  # Replace with your image path
check_image_compression(image_path)
