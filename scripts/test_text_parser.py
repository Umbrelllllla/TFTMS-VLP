"""自由文本解析器测试用例（自检）。

覆盖：数值区间、同义词、大小写、否定表达、复合颜色、距离近似表达。
用法：uv run python -m scripts.test_text_parser
"""

from methods.baseline.attributes import from_text

CASES = [
    # (描述, 期望约束子集)
    ("a car with a width ranging from 1.2 to 1.6 meters",
     {"objectname": "car", "width": (1.2, 1.6)}),
    ("a vehicle that is not occluded",
     {"objectname": "vehicle", "occlusion": "no occluded"}),
    ("a vehicle approximately 20 meters away",
     {"objectname": "vehicle", "distance": 20.0}),
    ("yellow-orange truck",
     {"appearance": "yellow|orange", "objectname": "truck"}),
    ("A Black CAR on the LEFT",
     {"appearance": "black", "objectname": "car", "place": "left"}),
    ("a gray sedan",
     {"appearance": "gray", "objectname": "car"}),
    ("a silver van in the bottom right corner",
     {"appearance": "gray", "objectname": "van", "place": "bottom right"}),
    ("a length between 4 and 5 meters",
     {"length": (4.0, 5.0)}),
    ("a person standing on the right",
     {"objectname": "pedestrian", "place": "right"}),
    ("the blue bus in the middle of the image",
     {"appearance": "blue", "objectname": "bus", "place": "center"}),
    ("roughly 15 m away",
     {"distance": 15.0}),
    ("side facing me",
     {"facing": "side facing me"}),
    ("facing away from me",
     {"facing": "facing away from me"}),
    ("a height of about 1.5 meters",
     {"height": (1.5, 1.5)}),
    ("partially occluded white car",
     {"occlusion": "partially occluded", "appearance": "white", "objectname": "car"}),
    ("heavily occluded cyclist",
     {"occlusion": "occluded", "objectname": "cyclist"}),
    # 真实模板句式（错误案例中发现的关键形态）
    ("Vehicles, which are cars, have a width ranging from approximately 1.5 to 1.9 meters.",
     {"objectname": "car", "width": (1.5, 1.9)}),
    ("Vehicles which are facing me, which have a height of approximately 0.9 to 1.3 meters.",
     {"objectname": "vehicle", "facing": "facing me", "height": (0.9, 1.3)}),
    ("Vehicles, which are located at the top left of the scene without any occlusion, have a length of approximately 4.2 to 4.6 meters.",
     {"objectname": "vehicle", "place": "top left", "occlusion": "no occluded", "length": (4.2, 4.6)}),
    ("Vehicles, which are not occluded and identified as cars, have a width ranging from approximately 1.9 to 2.3 meters.",
     {"objectname": "car", "occlusion": "no occluded", "width": (1.9, 2.3)}),
    # 2026-10-24 修复的复数形式 bug（错误案例中发现：lengths/heights/widths 漏解析）
    ("Vehicles, which are located within approximately 88.2 meters, and have lengths ranging from 4.1 to 4.5 meters.",
     {"distance": 88.2, "length": (4.1, 4.5)}),
    ("Vehicles which are facing me, whose heights range from approximately 1.3 to 1.7 meters, are present in the scene.",
     {"facing": "facing me", "height": (1.3, 1.7)}),
    ("Vehicles, which are cars facing me at a distance of approximately 96.6 meters, have lengths ranging from 4.2 to 4.6 meters.",
     {"objectname": "car", "facing": "facing me", "distance": 96.6, "length": (4.2, 4.6)}),
    ("Vehicles which have widths ranging from approximately 1.6 to 2.0 meters.",
     {"width": (1.6, 2.0)}),
]


def main() -> None:
    n_pass = 0
    for text, expect in CASES:
        c = from_text(text)
        ok = True
        detail = []
        for k, v in expect.items():
            got = getattr(c, k)
            if got != v:
                ok = False
                detail.append(f"  [{k}] 期望={v!r} 实际={got!r}")
        tag = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        print(f"{tag} | {text}")
        for d in detail:
            print(d)
    print(f"\n{ n_pass }/{len(CASES)} 通过")


if __name__ == "__main__":
    main()
