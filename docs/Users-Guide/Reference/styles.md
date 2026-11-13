---
title: Styles
icon: lucide/notepad-text
---

# Styles

The list of available built-in styles is:

```python exec="true"
from pathlib import Path

style_dir = Path("src/gusteau_translate/styles")

to_visit = {c:0 for c in sorted(style_dir.iterdir(),reverse=True)}
while to_visit:
    child, indent_level = to_visit.popitem()
    indent = "    " * indent_level
    if child.is_dir():
        print(f"{indent}{child.stem}/\n{indent}: ")
        to_visit.update(
            {
                c:indent_level + 1 
                for c in sorted(
                    child.iterdir(),
                    reverse=True
                )
            }
        )
        continue
    print(f"{indent}{child.stem}\n")
    # only want to read the first two lines
    text = child.read_text()
    lines = text.split("\n")
    print(f"{indent}:  ",)
    for line in lines:
        print(f"{indent}  {line[1:].strip()}<br>")
        if "version:" in line.lower():
            break
    print()
```

Styles are parsed with the following functions:

::: gusteau_translate.styles.parse_modular_style_file
    options:
        show_root_heading: true
        heading_level: 3

::: gusteau_translate.styles.parse_style_file
    options:
        show_root_heading: true
        heading_level: 3