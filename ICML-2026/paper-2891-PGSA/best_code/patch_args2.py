with open("/repo/main/args.py", "r") as f:
    content = f.read()

# Change epochs from 300 to 500 for dblp_acm
content = content.replace(
    "if (args.dataset or \"\").strip().lower() == \"dblp_acm\":\n        args.epochs = 300\n        args.h_threshold = 1.0\n        args.start_epoch = 200\n        args.rw_freq = 15",
    "if (args.dataset or \"\").strip().lower() == \"dblp_acm\":\n        args.epochs = 300\n        args.h_threshold = 1.0\n        args.start_epoch = 200\n        args.rw_freq = 15\n        args.opt_decay_step = 100"
)

with open("/repo/main/args.py", "w") as f:
    f.write(content)
print("args.py updated with opt_decay_step=100")
