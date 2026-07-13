#!/usr/bin/env python3
with open("/repo/train_NAVAR.py", "r") as f:
    content = f.read()

# Fix: move scheduler.step() from inside batch loop to after all batches
# Remove the per-batch scheduler step
old = """            optimizer.step()
            scheduler.step()

        # every 'check_every' epochs we calculate and print the validation loss"""
new = """            optimizer.step()

        # Step scheduler once per epoch (after all batches)
        scheduler.step()

        # every 'check_every' epochs we calculate and print the validation loss"""
content = content.replace(old, new)

with open("/repo/train_NAVAR.py", "w") as f:
    f.write(content)
print("Fixed scheduler: now steps once per epoch")
