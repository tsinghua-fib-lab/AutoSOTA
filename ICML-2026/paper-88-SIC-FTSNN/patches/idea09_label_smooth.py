with open("/repo/simple_snn.py", "r") as f:
    content = f.read()

# Soften one-hot targets
old = "        target_onehot = nn.functional.one_hot(targets, num_classes).float()"
new = "        target_onehot = nn.functional.one_hot(targets, num_classes).float()\n        # Label smoothing: 0.05 uniform noise\n        eps = 0.05\n        target_onehot = target_onehot * (1 - eps) + eps / num_classes"
content = content.replace(old, new)

with open("/repo/simple_snn.py", "w") as f:
    f.write(content)

print("IDEA-09 patch applied")
