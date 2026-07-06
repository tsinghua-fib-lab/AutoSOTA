with open("/repo/simple_snn.py", "r") as f:
    content = f.read()

# Add clip_grad_norm_ before optimizer.step()
old = "        optimizer.zero_grad()\n        loss_val.backward()\n        optimizer.step()"
new = "        optimizer.zero_grad()\n        loss_val.backward()\n        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)\n        optimizer.step()"
content = content.replace(old, new)

with open("/repo/simple_snn.py", "w") as f:
    f.write(content)

print("IDEA-08 patch applied")
