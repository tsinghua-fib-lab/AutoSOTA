with open("/repo/main/args.py", "r") as f:
    content = f.read()

# Insert paper settings
old_return = "    if (args.dataset or \"\").strip().lower() == \"noncircle\":\n        args.dataset = \"Noncircle\"\n    return args"
new_return = "    if (args.dataset or \"\").strip().lower() == \"noncircle\":\n        args.dataset = \"Noncircle\"\n    # Paper settings for dblp_acm (Table 6, Appendix B.3)\n    if (args.dataset or \"\").strip().lower() == \"dblp_acm\":\n        args.epochs = 300\n        args.h_threshold = 1.0\n        args.start_epoch = 200\n        args.rw_freq = 15\n    return args"

if old_return in content:
    content = content.replace(old_return, new_return)
    with open("/repo/main/args.py", "w") as f:
        f.write(content)
    print("Paper settings restored")
else:
    print("Could not find old return statement")
