# src/test_data_loader.py
from data_loader import get_non_iid_data, get_test_data

def test_data_loading():
    uav_id = 0
    total_uavs = 6
    train_loader, val_loader = get_non_iid_data(uav_id, total_uavs)
    test_loader = get_test_data()

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

if __name__ == "__main__":
    test_data_loading()