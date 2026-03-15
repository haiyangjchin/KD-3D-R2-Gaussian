#!/usr/bin/env python3
"""
Evaluate seashell training results for all iterations including 10000.
"""

import sys

sys.path.append(".")

from evaluate_distill_results import evaluate_all_iterations, save_results_to_csv


def main():
    # Evaluate seashell training
    base_path = "./distill_student_10k_seashell_tv"
    print(f"Evaluating seashell training results from: {base_path}")

    results = evaluate_all_iterations(base_path=base_path)

    # Save results with seashell-specific filename
    if results:
        save_results_to_csv(
            results, output_path="distill_evaluation_results_seashell_tv.csv"
        )
        print("\nSeashell evaluation completed!")

        # Print summary
        print("\nFinal results:")
        for iter_name, metrics in sorted(results.items()):
            print(
                f"{iter_name}: MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}, "
                f"PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.6f}"
            )
    else:
        print("No results were generated.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
