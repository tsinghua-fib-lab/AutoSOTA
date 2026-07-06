with open("/repo/simple_snn.py", "r") as f:
    content = f.read()

# Replace ATan() with Sigmoid(alpha=4.0)
content = content.replace("surrogate_function=surrogate.ATan()", "surrogate_function=surrogate.Sigmoid(alpha=4.0)")

with open("/repo/simple_snn.py", "w") as f:
    f.write(content)

print("IDEA-05 patch applied")
