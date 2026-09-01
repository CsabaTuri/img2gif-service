import base64
import os
from app import service

def main():
    # Helyezz el egy 'input.jpg' fájlt a projekt mappájába a teszthez
    image_path = "input.jpg"
    
    if not os.path.exists(image_path):
        print(f"[HIBA] Kérlek tegyél egy '{image_path}' képet ebbe a mappába a teszteléshez!")
        return

    print("1. Tesztkép beolvasása és átalakítása Base64 formátumba...")
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    print("2. Generálás indítása a futó Ollama és lokális Diffusers segítségével...")
    gif_b64 = service.generate_gif_from_image(
        image_b64=img_b64,
        instruction="Make the background dynamic with flowing camera move",
        num_frames=16
    )

    output_filename = "result.gif"
    with open(output_filename, "wb") as f:
        f.write(base64.b64decode(gif_b64))

    print(f"3. SIKER! A generált GIF elmentve: {output_filename}")

if __name__ == "__main__":
    main()