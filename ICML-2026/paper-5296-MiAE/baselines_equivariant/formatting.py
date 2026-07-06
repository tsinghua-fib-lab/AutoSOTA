def format_table(headers, rows, right_align=None):
    right_align = set(right_align or [])
    rows = [[str(value) for value in row] for row in rows]
    headers = [str(header) for header in headers]

    widths = [
        max(len(header), *(len(row[idx]) for row in rows)) if rows else len(header)
        for idx, header in enumerate(headers)
    ]

    def format_row(row):
        cells = []
        for idx, value in enumerate(row):
            if idx in right_align:
                cells.append(value.rjust(widths[idx]))
            else:
                cells.append(value.ljust(widths[idx]))
        return "  ".join(cells)

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in rows)])
