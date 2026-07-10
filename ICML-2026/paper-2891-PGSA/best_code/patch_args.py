# Patch args.py for paper settings
with open("/repo/main/args.py", "r") as f:
    content = f.read()

# Replace function to add paper settings
old = """def apply_psahs_defaults(args):
    \"\"\"Apply tuned defaults used in the paper implementation.\"\"\"
    args.reweight = \"rw\" in (args.method or \"\")
    if args.reweight:
        args.rw_freq = 20
        args.start_epoch = 0
    args.lr = 0.003
    args.opt_decay_rate = 0.8
    args.opt_decay_step = 50
    args.hidden_dim = 128
    args.conv_dim = 128
    args.cls_dim = 64
    args.alphamin = 1.0
    args.alphatimes = 1.5
    args.epochs = 200
    args.h_threshold = 0.6
    args.K = 2
    if (args.dataset or \"\").strip().lower() == \"noncircle\":
        args.dataset = \"Noncircle\"
    return args"""

new = """def apply_psahs_defaults(args):
    \"\"\"Apply tuned defaults used in the paper implementation.\"\"\"
    args.reweight = \"rw\" in (args.method or \"\")
    if args.reweight:
        args.rw_freq = 20
        args.start_epoch = 0
    args.lr = 0.003
    args.opt_decay_rate = 0.8
    args.opt_decay_step = 50
    args.hidden_dim = 128
    args.conv_dim = 128
    args.cls_dim = 64
    args.alphamin = 1.0
    args.alphatimes = 1.5
    args.epochs = 200
    args.h_threshold = 0.6
    args.K = 2
    if (args.dataset or \"\").strip().lower() == \"noncircle\":
        args.dataset = \"Noncircle\"
    # Paper settings for dblp_acm (Table 6, Appendix B.3)
    if (args.dataset or \"\").strip().lower() == \"dblp_acm\":
        args.epochs = 300
        args.h_threshold = 1.0
        args.start_epoch = 200
        args.rw_freq = 15
    return args"""

if old in content:
    content = content.replace(old, new)
    with open("/repo/main/args.py", "w") as f:
        f.write(content)
    print("args.py patched successfully")
else:
    print("ERROR: Could not find function to patch")
