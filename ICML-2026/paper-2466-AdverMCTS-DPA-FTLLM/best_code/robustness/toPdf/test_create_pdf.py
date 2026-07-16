from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Output PDF file
output_file = "char_all.pdf"

# Register a Unicode TTF font (DejaVuSans supports a wide range of glyphs)
pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))

# Create PDF canvas
c = canvas.Canvas(output_file, pagesize=A4)
width, height = A4
c.setFont("DejaVuSans", 12)

# Define all codepoints explicitly
codepoints = [
    0x061C, 0x180E,
    0x200B, 0x200C, 0x200D, 0x200E, 0x200F,
    0x202A, 0x202C, 0x202D,
    0x2060, 0x2061, 0x2062, 0x2063, 0x2064,
    0x2066, 0x2068, 0x2069,
    0x206A, 0x206B, 0x206C, 0x206D, 0x206E, 0x206F,
    0xFEFF,
    0x1D173, 0x1D174, 0x1D175, 0x1D176, 0x1D177, 0x1D178, 0x1D179, 0x1D17A,
    0xE0001,
] + list(range(0xE0020, 0xE0080))  # tag characters U+E0020–U+E007F

# Construct string
text = "A" + "".join(chr(cp) for cp in codepoints) + "B"

# Draw text (invisible characters will render as blank space)
c.drawString(100, 700, text)

# Save the PDF
c.save()

print(f"PDF created successfully with {len(codepoints)} invisible characters!")


