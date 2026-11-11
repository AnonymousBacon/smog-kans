# old name for run_configs.py, kept until the daniel comparison batch is done; then delete it.
import sys

from run_configs import SUMMARY_COLS, load_runs, main, write_summary

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    main()
