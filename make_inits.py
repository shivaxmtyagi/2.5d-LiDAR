from pathlib import Path

root = Path("src")
for d in [root, *root.rglob("*")]:
    if d.is_dir() and d.name != "__pycache__":
        (d / "__init__.py").touch(exist_ok=True)
        print("ok:", d / "__init__.py")