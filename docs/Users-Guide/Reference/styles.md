---
title: Styles
icon: lucide/notepad-text
---

# Styles

The list of available built-in styles is:

```python exec="true"
from pathlib import Path

style_dir = Path("src/gusteau_translate/styles")

for child in style_dir.iterdir():
    print(f"{child.stem}\n")
    # only want to read the first two lines
    text = child.read_text()
    lines = text.split("\n")
    print(":", end="")
    for line in lines:
        print(f"  {line[1:].strip()}<br>")
        if "version" in line.lower():
            break
    print()
```