def main():
    import sys
    print("trainable_ot_dro imported successfully.")
    try:
        from . import bilevel_optimization
        print("Core modules available.")
    except Exception as e:
        print("Import check failed:", e)
        sys.exit(1)  # non-zero exit on failure
