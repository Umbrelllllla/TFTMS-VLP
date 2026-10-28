"""验证 from_text 尺寸解析：查真实错误案例的描述能否正确解析出 height/width/length。"""

import os

from methods.baseline.attributes import from_text

proj = r"D:\TFTMS-VLP"
# 从错误案例中取真实描述
descs = [
    "Vehicles which are facing me, whose heights range from approximately 1.3 to 1.7 meters, are present in the scene.",
    "Vehicles, which are located within approximately 95 meters, have a height ranging from 1.2 to 1.6 meters.",
    "Vehicles, which are located within approximately 50 meters, have a height ranging from 1.1 to 1.5 meters.",
    "Vehicles, which are located on the top and partially occluded, and identified as cars, have a length ranging approximately from 4.0 to 4.4 meters.",
    "Vehicles, which are cars and are not occluded, are located within approximately 88.2 meters, and have lengths ranging from 4.1 to 4.5 meters.",
    "Vehicles, which are cars facing me at a distance of approximately 96.6 meters, have lengths ranging from 4.2 to 4.6 meters.",
    "Vehicles, which are located in the top right of the scene and have a white appearance, are approximately 1.1 to 1.5 meters in height.",
    "Vehicles, which are black cars, have a length of approximately 4.0 to 4.4 meters.",
    "Vehicles, which are facing away from me and have a silver-grey appearance, are cars with a length of approximately 4.0 to 4.4 meters.",
    "Vehicles, which have a silver-grey appearance, are approximately 1.1 to 1.5 meters in height.",
]

for d in descs:
    c = from_text(d)
    print(f"desc: {d[:90]}")
    print(f"  → height={c.height} width={c.width} length={c.length} dist={c.distance} "
          f"place={c.place} facing={c.facing} color={c.appearance} occ={c.occlusion}")
