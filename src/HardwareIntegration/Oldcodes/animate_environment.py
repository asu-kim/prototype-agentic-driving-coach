import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "environment.csv"
OUTPUT_PATH = ROOT / "environment_road_animation.gif"

WIDTH, HEIGHT = 900, 600
FRAME_STEP = 5
FRAME_DURATION_MS = 100 * FRAME_STEP

GRASS = "#6f9f54"
ROAD = "#44484d"
LANE = "#f5f1cf"
EGO = "#32a8ff"
TRAFFIC = "#ef5350"


def text(draw, xy, value, fill="white", anchor=None):
    draw.text(xy, value, font=ImageFont.load_default(), fill=fill, anchor=anchor)


def car(draw, x, y, color, label, angle=0):
    car_image = Image.new("RGBA", (70, 120), (0, 0, 0, 0))
    car_draw = ImageDraw.Draw(car_image)
    car_draw.rounded_rectangle((10, 5, 60, 115), radius=10, fill=color, outline="white", width=3)
    car_draw.rectangle((17, 28, 53, 53), fill="#bde7ff")
    car_draw.rectangle((17, 70, 53, 95), fill="#bde7ff")
    text(car_draw, (35, 61), label, fill="black", anchor="mm")
    if angle:
        car_image = car_image.rotate(angle, expand=True)
    frame.alpha_composite(car_image, (int(x - car_image.width / 2), int(y - car_image.height / 2)))


def draw_status(draw, row, second):
    draw.rectangle((0, 0, WIDTH, 76), fill="#18202a")
    text(draw, (20, 18), f"Environment scenario: {row['road_phase']}", anchor="lm")
    text(draw, (20, 46), f"Time: {second:05.1f} / 85.0 seconds", fill="#c7d4e2", anchor="lm")
    progress = min(second / 85.0, 1.0)
    draw.rounded_rectangle((310, 25, 860, 45), radius=7, fill="#3d4855")
    draw.rounded_rectangle((310, 25, 310 + 550 * progress, 45), radius=7, fill="#62c370")


def draw_highway(draw, row, index):
    draw.rectangle((210, 76, 690, HEIGHT), fill=ROAD)
    draw.line((450, 76, 450, HEIGHT), fill=LANE, width=5)
    for y in range(90, HEIGHT, 55):
        draw.line((330, y, 330, y + 28), fill=LANE, width=4)
        draw.line((570, y, 570, y + 28), fill=LANE, width=4)

    phase = row["road_phase"]
    ego_x = 390
    if phase == "LANE_CHANGE":
        progress = min(max((index - 300) / 100, 0.0), 1.0)
        ego_x = 390 + 120 * progress
    elif phase == "EXIT":
        ego_x = 510
        draw.polygon(((450, 300), (690, 300), (900, 500), (900, 600), (690, 430)), fill=ROAD)
        draw.line((570, 315, 860, 565), fill=LANE, width=5)

    car(draw, ego_x, 430, EGO, "EGO")

    if row["front_car_present"] == "1":
        car(draw, ego_x, 220, TRAFFIC, "FRONT")

    if row["right_car_present"] == "1":
        positions = {"RIGHT_REAR": 520, "RIGHT_SIDE": 430, "RIGHT_FRONT": 250}
        car(draw, 510, positions.get(row["right_car_position"], 430), "#ff9f43", "RIGHT")


def draw_intersection(draw, row, phase):
    draw.rectangle((330, 76, 570, HEIGHT), fill=ROAD)
    draw.rectangle((0, 225, WIDTH, 425), fill=ROAD)
    draw.line((450, 76, 450, 205), fill=LANE, width=5)
    draw.line((450, 445, 450, HEIGHT), fill=LANE, width=5)
    draw.line((0, 325, 310, 325), fill=LANE, width=5)
    draw.line((590, 325, WIDTH, 325), fill=LANE, width=5)
    draw.line((330, 455, 570, 455), fill="white", width=10)

    draw.ellipse((590, 425, 650, 485), fill="#d92626", outline="white", width=4)
    text(draw, (620, 455), "STOP", anchor="mm")

    ego_y = 500 if phase in ("STOP_SIGN", "YIELD") else 350
    car(draw, 450, ego_y, EGO, "EGO")

    if row["cross_left_car_present"] == "1":
        car(draw, 185, 325, TRAFFIC, "LEFT", angle=90)
    if row["cross_right_car_present"] == "1":
        car(draw, 715, 325, "#ff9f43", "RIGHT", angle=90)


with CSV_PATH.open(newline="") as file:
    rows = list(csv.DictReader(file))

frames = []
for index in range(0, len(rows), FRAME_STEP):
    row = rows[index]
    frame = Image.new("RGBA", (WIDTH, HEIGHT), GRASS)
    draw = ImageDraw.Draw(frame)
    draw_status(draw, row, index * 0.1)

    if row["road_phase"] in ("HIGHWAY", "LANE_CHANGE", "EXIT"):
        draw_highway(draw, row, index)
    else:
        draw_intersection(draw, row, row["road_phase"])

    frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE))

frames[0].save(
    OUTPUT_PATH,
    save_all=True,
    append_images=frames[1:],
    duration=FRAME_DURATION_MS,
    loop=0,
    optimize=True,
)
print(f"Created {OUTPUT_PATH} with {len(frames)} frames.")
