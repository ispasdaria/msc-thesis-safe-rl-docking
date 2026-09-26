from pathlib import Path


CONTROLLERS_ROOT = Path(__file__).resolve().parents[1]


def test_controller_scripts_do_not_generate_pdf_files():
    offenders = []
    for path in CONTROLLERS_ROOT.rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        if ".pdf" in path.read_text(encoding="utf-8").lower():
            offenders.append(path.relative_to(CONTROLLERS_ROOT).as_posix())

    assert offenders == []
