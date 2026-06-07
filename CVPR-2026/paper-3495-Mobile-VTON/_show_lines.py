with open('/repo/inference.py') as f:
    lines = f.readlines()
for i in range(306, 326):
    print(repr(lines[i]))
with open('/repo/_iter3_lines.txt', 'w') as out:
    for i in range(306, 326):
        out.write(f'{i+1}: {lines[i]}')
print('OK')
