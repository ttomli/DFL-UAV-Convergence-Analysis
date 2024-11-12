# src/test_uav.py
from uav import UAV
from model import SimpleCNN

def test_uav():
    model = SimpleCNN()
    position = [250, 250]
    uav = UAV(uav_id=0, position=position, model=model)
    print(f"Initial position: {uav.position}")

    uav.move()
    print(f"Position after movement: {uav.position}")

if __name__ == "__main__":
    test_uav()