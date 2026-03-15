#!/usr/bin/env python3
"""
Merge original training log (0-9630) with resume log (7500-10000)
to create a complete training log from 0 to 10000 iterations.
"""

import os


def main():
    original_log = "distill_10k_seashell_tv_training.log"
    resume_log = "distill_10k_seashell_tv_resume_final.log"
    merged_log = "distill_10k_seashell_tv_complete.log"

    # Backup original log
    backup_log = original_log + ".backup"
    if not os.path.exists(backup_log):
        print(f"Backing up original log to {backup_log}")
        with open(original_log, "rb") as f_in, open(backup_log, "wb") as f_out:
            f_out.write(f_in.read())

    # Read original log (binary mode)
    print(f"Reading original log: {original_log}")
    with open(original_log, "rb") as f:
        original_data = f.read()

    # Read resume log (binary mode)
    print(f"Reading resume log: {resume_log}")
    with open(resume_log, "rb") as f:
        resume_data = f.read()

    # Find the position of "Resuming from iteration 7500" in resume log
    # Convert to bytes for search
    search_str = b"Resuming from iteration 7500"
    pos = resume_data.find(search_str)

    if pos == -1:
        print("ERROR: Could not find 'Resuming from iteration 7500' in resume log")
        print("Trying alternative search...")
        # Try other possible patterns
        search_strs = [
            b"Resuming from iteration",
            b"Resuming Training from Pickle",
            b"Resumed from iteration: 7500",
        ]
        for s in search_strs:
            pos = resume_data.find(s)
            if pos != -1:
                print(f"Found alternative pattern: {s}")
                break

        if pos == -1:
            print("ERROR: No resume point found. Using entire resume log.")
            resume_start = 0
        else:
            # Find the beginning of this line
            line_start = resume_data.rfind(b"\n", 0, pos)
            if line_start == -1:
                line_start = 0
            else:
                line_start += 1  # Skip the newline
            resume_start = line_start
    else:
        # Find the beginning of this line
        line_start = resume_data.rfind(b"\n", 0, pos)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1  # Skip the newline
        resume_start = line_start

    print(f"Resume point found at byte position: {resume_start}")

    # Check if resume log contains "Training completed!"
    if b"Training completed!" in resume_data:
        print(
            "Resume log contains 'Training completed!' - training finished successfully."
        )

    # Create merged log
    print(f"Creating merged log: {merged_log}")
    with open(merged_log, "wb") as f:
        # Write original log
        f.write(original_data)
        # Add a separator
        f.write(b"\n" + b"=" * 80 + b"\n")
        f.write(
            b"TRAINING RESUMED FROM ITERATION 7500 (previous training interrupted at iteration 9630)\n"
        )
        f.write(b"=" * 80 + b"\n\n")
        # Write resume log from the resume point
        f.write(resume_data[resume_start:])

    print(f"Merged log created: {merged_log}")
    print(
        f"Size: original={len(original_data)}, resume={len(resume_data)}, merged={len(original_data) + len(resume_data[resume_start:])}"
    )

    # Optional: Replace the original log with merged log
    # But we'll keep original as backup and create new complete log
    # If user wants to overwrite original, they can do it manually

    print("\nDone!")
    print(f"Original log backup: {backup_log}")
    print(f"Complete training log: {merged_log}")
    print("You can rename merged log to original log if desired.")


if __name__ == "__main__":
    main()
