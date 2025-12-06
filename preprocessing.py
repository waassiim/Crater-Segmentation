import pickle
import multiprocessing
import math
import random
import os
import numpy as np
import matplotlib.image
from PIL import Image
from patchify import patchify, unpatchify

# Configuration
source_dir = "data"
CHECKPOINT_FILE = "processing_checkpoint_v2.pkl"

# Ensure output directory exists
if not os.path.exists(source_dir):
    os.makedirs(source_dir)


def reconstruct_images(patches):
    # Hardcoded size based on previous context (7680x7680)
    reconstructed_image = unpatchify(patches, (7680, 7680))
    return reconstructed_image


def get_image_source_list(source):
    l = os.listdir(source)
    return l


def return_slised_image(large_image):
    patches = patchify(large_image, (512, 512), step=512)
    return patches


def convert_grey(img):
    return img.convert("L")


def get_pixels(images):
    """
    Returns a list of (x, y) tuples where x=col, y=row for pixels == 0.
    """
    l = []
    # Standard traversal: y (row) then x (col) is usually faster in numpy,
    # but order doesn't impact logic as long as access is consistent.
    for x in range(512):
        for y in range(512):
            # FIXED: Access numpy array as [row][col] -> [y][x]
            pixel_val = images[y][x]
            if pixel_val == 0:
                l.append((x, y))
    return l


def get_neighboring_pixel(img, x, y, current_window_size):
    """
    Inputs: img (numpy array [row][col]), x (col), y (row)
    """
    x_rand, y_rand = 0, 0
    max_num_tries = 10000
    max_tries_per_neighbourhood = 50
    neighbourhood_size_increment = 50

    # Calculate how many window expansions we might need
    num_expansions = math.ceil(max_num_tries / max_tries_per_neighbourhood)

    for _ in range(num_expansions):
        for _ in range(max_tries_per_neighbourhood):
            min_x = max(0, x - current_window_size)
            max_x = min(512, x + current_window_size)
            min_y = max(0, y - current_window_size)
            max_y = min(512, y + current_window_size)

            # randint is inclusive for both bounds? No, python randint is [a, b].
            # However, numpy indices are 0-511.
            # If max_x is 512, we want randint to produce up to 511.
            # random.randint(a, b) produces N where a <= N <= b.
            # So we use max_x - 1.
            x_rand = random.randint(min_x, max_x - 1)
            y_rand = random.randint(min_y, max_y - 1)

            # FIXED: Access numpy array as [row][col] -> [y][x]
            if not img[y_rand][x_rand] == 0:
                return x_rand, y_rand
        current_window_size += neighbourhood_size_increment

    return x_rand, y_rand


def fill_swath_with_neighboring_pixel(img, left=10, right=100, top=10, bottom=100, color=(0, 0, 0),
                                      current_window_size=10):
    img_with_neighbor_filled = np.array(img.copy())
    l = get_pixels(img)

    for k in l:
        x, y = k  # x is col, y is row
        x_rand, y_rand = get_neighboring_pixel(img, x, y, current_window_size)

        # Check boundaries (coordinates are x=col, y=row)
        if x >= left and x <= right and y >= top and y <= bottom:
            # FIXED: Access numpy array as [row][col] -> [y][x]
            img_with_neighbor_filled[y][x] = img[y_rand][x_rand]

    return img_with_neighbor_filled


# ---------------------------------------------------------
# 1. WORKER FUNCTION
# ---------------------------------------------------------
def process_single_patch(task_data):
    i, j, patch = task_data

    processed_patch = fill_swath_with_neighboring_pixel(
        patch,
        left=0, right=511, top=0, bottom=511,
        color=(0, 0, 0),  # Fixed set {} to tuple ()
        current_window_size=10
    )

    return (i, j, processed_patch)


# ---------------------------------------------------------
# 2. CHECKPOINT MANAGERS
# ---------------------------------------------------------
def save_checkpoint(filename, patches, completed_indices):
    data = {
        "filename": filename,
        "patches": patches,
        "completed_indices": completed_indices
    }
    temp_file = CHECKPOINT_FILE + ".tmp"
    with open(temp_file, "wb") as f:
        pickle.dump(data, f)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    os.rename(temp_file, CHECKPOINT_FILE)


def load_checkpoint(current_filename):
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                data = pickle.load(f)
                if data["filename"] == current_filename:
                    return data
        except Exception as e:
            print(f"Checkpoint corrupt or unreadable, starting fresh: {e}")
            return None
    return None


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint cleared.")


# ---------------------------------------------------------
# 3. MAIN PROCESSING CONTROLLER
# ---------------------------------------------------------
def main_processing_loop():
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting Multiprocessing with {num_cores} workers.")

    for img_name in image_files:
        input_path = os.path.join(source_dir, img_name)
        output_path = os.path.join(source_dir, img_name + "_good_image.png")

        if os.path.exists(output_path):
            print(f"Skipping {img_name}, output already exists.")
            continue

        print(f"Preparing: {img_name}...")

        checkpoint_data = load_checkpoint(img_name)
        patches = None
        completed_indices = set()

        if checkpoint_data:
            print(
                f"Found checkpoint! Resuming {img_name} with {len(checkpoint_data['completed_indices'])} patches already done.")
            patches = checkpoint_data['patches']
            completed_indices = checkpoint_data['completed_indices']
        else:
            try:
                large_image = Image.open(input_path)
                im = convert_grey(large_image)
                im_array = np.array(im)

                if im_array.shape[0] != 7680 or im_array.shape[1] != 7680:
                    print(f"Resizing {img_name} to 7680x7680...")
                    im = im.resize((7680, 7680))
                    im_array = np.array(im)

                patches = return_slised_image(im_array)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
                continue

        rows = patches.shape[0]
        cols = patches.shape[1]

        tasks = []
        for i in range(rows):
            for j in range(cols):
                if (i, j) not in completed_indices:
                    tasks.append((i, j, patches[i][j]))

        total_tasks = rows * cols
        remaining_tasks = len(tasks)
        print(f"Processing {remaining_tasks} remaining patches out of {total_tasks} total...")

        if remaining_tasks > 0:
            with multiprocessing.Pool(processes=num_cores) as pool:
                for result in pool.imap_unordered(process_single_patch, tasks):
                    r_i, r_j, processed_patch_data = result

                    patches[r_i][r_j] = processed_patch_data
                    completed_indices.add((r_i, r_j))

                    # MODIFICATION: Only save checkpoint every 10 patches to reduce Disk I/O overhead
                    if len(completed_indices) % 10 == 0:
                        save_checkpoint(img_name, patches, completed_indices)
                        print(f"Checkpoint saved. Progress: {len(completed_indices)}/{total_tasks}")
                    else:
                        print(f"Finished patch ({r_i}, {r_j}) - Progress: {len(completed_indices)}/{total_tasks}")

        print(f"Reconstructing and saving {img_name}...")
        try:
            good_image = reconstruct_images(patches)
            matplotlib.image.imsave(output_path, good_image)
            print(f"Saved: {output_path}")
            clear_checkpoint()
        except Exception as e:
            print(f"Failed to save {img_name}: {e}")

    print("All images processed.")


if __name__ == '__main__':
    main_processing_loop()