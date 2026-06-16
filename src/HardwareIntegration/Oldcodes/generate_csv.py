import csv
import os

def generate_environment_csv():
    filepath = os.path.join(os.path.dirname(__file__), "environment.csv")
    rows = []
    dt = 0.1  # LF reactor reads every 100ms
    
    # Driving speeds in meters per second
    speed_hwy = 120.0 / 3.6  # 33.33 m/s
    speed_exit = 40.0 / 3.6  # 11.11 m/s
    
    # Track the cumulative distance for synchronization
    current_distance = 0.0
    
    def add_rows(distance, speed, phase, front, right, right_pos, left_c, right_c):
        nonlocal current_distance
        duration_sec = distance / speed
        num_rows = int(duration_sec / dt)
        for _ in range(num_rows):
            rows.append([phase, front, right, right_pos, left_c, right_c])
            rows.append([round(current_distance, 2), phase, front, right, right_pos, left_c, right_c])
            current_distance += (speed * dt)
            
    def add_time(duration_sec, phase, front, right, right_pos, left_c, right_c):
        nonlocal current_distance
        num_rows = int(duration_sec / dt)
        for _ in range(num_rows):
            rows.append([phase, front, right, right_pos, left_c, right_c])
            rows.append([round(current_distance, 2), phase, front, right, right_pos, left_c, right_c])

    # --- HIGHWAY PHASE (0 to 4000 meters) ---
    # 0 to 2000m: Clear highway cruising
    add_rows(2000, speed_hwy, "HIGHWAY", 0, 0, "NONE", 0, 0)
    # 2000 to 2500m: Front car appears ahead
    add_rows(500, speed_hwy, "HIGHWAY", 1, 0, "NONE", 0, 0)
    
    # 2500 to 3500m: Lane Change Phase (Passes the sign at 3000m)
    add_rows(300, speed_hwy, "LANE_CHANGE", 0, 1, "RIGHT_REAR", 0, 0)
    add_rows(400, speed_hwy, "LANE_CHANGE", 0, 1, "RIGHT_SIDE", 0, 0)
    add_rows(300, speed_hwy, "LANE_CHANGE", 0, 1, "RIGHT_FRONT", 0, 0)
    
    # 3500 to 4000m: Right car speeds away, clear to take exit
    add_rows(500, speed_hwy, "LANE_CHANGE", 0, 0, "NONE", 0, 0)

    # --- EXIT PHASE (4000m to 5212m equivalent length) ---
    # Drive 612m on exit road to the Speed Limit 40 sign
    add_rows(612, speed_exit, "EXIT", 0, 0, "NONE", 0, 0)
    # Drive next 488m on exit road with a car in front
    add_rows(488, speed_exit, "EXIT", 1, 0, "NONE", 0, 0)
    # Final 112m approaching the Stop Sign
    add_rows(112, speed_exit, "STOP_SIGN", 0, 0, "NONE", 0, 0)

    # --- INTERSECTION PHASE (Stopped) ---
    add_time(2.0, "STOP_SIGN", 0, 0, "NONE", 0, 0) # Wait 2 seconds
    add_time(3.0, "STOP_SIGN", 0, 0, "NONE", 1, 0) # Left red car approaches (3s)
    add_time(3.0, "STOP_SIGN", 0, 0, "NONE", 1, 1) # Right yellow car arrives (3s)
    add_time(3.0, "YIELD",     0, 0, "NONE", 0, 1) # Left car leaves, yield to right (3s)
    add_time(5.0, "MOVE",      0, 0, "NONE", 0, 0) # All clear, drive through (5s)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["road_phase", "front_car_present", "right_car_present", "right_car_position", "cross_left_car_present", "cross_right_car_present"])
        writer.writerow(["expected_distance_m", "road_phase", "front_car_present", "right_car_present", "right_car_position", "cross_left_car_present", "cross_right_car_present"])
        writer.writerows(rows)
        
if __name__ == "__main__":
    generate_environment_csv()