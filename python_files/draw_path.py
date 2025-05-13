from PIL import Image, ImageDraw


def draw_circle_on_image(image_path, output_path, center, radius, color=(255, 0, 0)):
    """
    Tekent een cirkel op een afbeelding.

    :param image_path: Pad naar de invoerafbeelding.
    :param output_path: Pad om de gewijzigde afbeelding op te slaan.
    :param center: Tuple (x, y) voor het middelpunt van de cirkel.
    :param radius: Straal van de cirkel.
    :param color: Kleur van de cirkel in (R, G, B) formaat.
    """
    # Open de afbeelding
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Definieer de bounding box van de cirkel
    x, y = center
    bbox = [(x - radius, y - radius), (x + radius, y + radius)]

    # Teken de cirkel
    draw.ellipse(bbox, fill=color, outline=color)

    # Sla de gewijzigde afbeelding op
    image.save(output_path)
    print(f"Afbeelding opgeslagen als {output_path}")


#Voorbeeldgebruik
with open("./../output/coordinates.txt", "r") as bestand:
    eerste_regel = True
    for regel in bestand:
        rij, kolom = map(int, regel.split())  # Splits en zet om naar int
        if eerste_regel == True:
            draw_circle_on_image("./../frames/vid_shorter_frames_png/0.png", "./../output/output_image.png", center=(kolom, rij), radius=8, color=(0, 255, 0)) #x = breedte = kolom, y = hoogte = rij
            eerste_regel = False
        else:
            draw_circle_on_image("./../output/output_image.png", "./../output/output_image.png", center=(kolom, rij), radius=8,color=(255, 255, 0))  # x = breedte = kolom, y = hoogte = rij


            