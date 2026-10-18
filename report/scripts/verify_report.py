"""Check saved evidence, report values and the compiled PDF."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

REPORT = Path(__file__).resolve().parents[1]
A4 = REPORT.parent
ROOT = A4.parent
OUT = REPORT / "generated"


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_arguments(source, command):
    """Read nested braced arguments without treating mathematical braces as ends."""
    arguments = []
    for match in re.finditer(r"\\" + command + r"\s*\{", source):
        start = match.end()
        depth = 1
        position = start
        while depth and position < len(source):
            char = source[position]
            if char == "{" and source[position - 1] != "\\":
                depth += 1
            elif char == "}" and source[position - 1] != "\\":
                depth -= 1
            position += 1
        assert depth == 0, f"Unbalanced argument for {command}"
        arguments.append(source[start : position - 1])
    return arguments


def narrative(source):
    """Strip typesetting and mathematics for conservative word-level checks."""
    text = re.sub(r"(?m)(?<!\\)%.*$", "", source)
    for environment in ["figure", "table", "list"]:
        text = re.sub(
            r"\\begin\{" + environment + r"\*?\}.*?\\end\{" + environment + r"\*?\}",
            "\n\n",
            text,
            flags=re.S,
        )
    text = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", text, flags=re.S)
    text = re.sub(r"\$[^$]*\$", " ", text)
    for command in [
        "label",
        "ref",
        "cite",
        "input",
        "includegraphics",
        "section",
        "subsection",
    ]:
        text = re.sub(r"\\" + command + r"(?:\[[^]]*\])?\{[^}]*\}", " ", text)
    return re.sub(r"\\[A-Za-z]+\*?", " ", text).replace("~", " ")


def main():
    manifest = json.loads((OUT / "asset_manifest.json").read_text())
    for path, expected in manifest["inputs"].items():
        assert fingerprint(A4 / path) == expected, f"Frozen input changed: {path}"
    for path, expected in manifest["assets"].items():
        assert fingerprint(OUT / path) == expected, f"Generated asset changed: {path}"
    final = pd.read_csv(
        A4 / "results/exploratory/chunk4_6/nested_summary.csv", float_precision="round_trip"
    )
    claims = (OUT / "claims.tex").read_text()
    for name, method in [
        ("Confirm", "k40"),
        ("ExploreA", "k150"),
        ("ExploreB", "k400"),
        ("Neighbour", "knn"),
        ("Tree", "tree"),
    ]:
        for suffix, metric in [
            ("F", "macro_f1"),
            ("B", "balanced_accuracy"),
            ("A", "accuracy"),
            ("W", "weighted_f1"),
        ]:
            value = final.loc[
                (final.method == method) & (final.metric == metric), "mean"
            ].item()
            assert f"\\newcommand{{\\{name}{suffix}}}{{{value:.3f}}}" in claims
    for source_name, copy_name in [
        ("results/som-v3/sweep_summary.csv", "representation_values"),
        ("results/exploratory/chunk4_5b/combined_validation_summary.csv", "sensitivity_values"),
        ("results/exploratory/chunk4_6/nested_summary.csv", "assessment_values"),
        ("results/exploratory/chunk4_6/per_class_summary.csv", "class_values"),
        ("results/exploratory/chunk4_6/bootstrap_intervals.csv", "bootstrap_values"),
        ("results/assessment/som_diagnostics.csv", "geometry_values"),
        ("results/exploratory/chunk4_6/fragmentation_summary.csv", "fragmentation_values"),
        ("results/exploratory/chunk4_6/structural_metrics.csv", "structural_values"),
        ("results/exploratory/chunk4_6/development_assessment_transfer.csv", "transfer_values"),
    ]:
        source_frame = pd.read_csv(A4 / source_name, float_precision="round_trip")
        display_frame = pd.read_csv(OUT / f"{copy_name}.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(source_frame, display_frame, check_exact=True)

    main_source = (REPORT / "main.tex").read_text()
    assert "conference,10pt,letterpaper" in main_source
    assert r"\IEEEauthorblockN{27284808 \quad 18 October 2026}" in main_source
    assert "hyperref" not in main_source
    assert r"\href" not in main_source
    sections = "\n".join(p.read_text() for p in (REPORT / "sections").glob("*.tex"))
    assert not re.search(r"TODO|FIXME|INSERT RESULTS|PLACEHOLDER", sections)
    assert r"\href" not in sections and "hyperref" not in sections
    prose = narrative(sections)
    first_person = re.findall(r"\b(?:we|our|ours|us|my|mine)\b|\bI\b", prose, re.I)
    assert not first_person, f"First person: {first_person}"
    forbidden = re.findall(r"\b(?:some|certain|etc)\b|\b(?:may|can) be\b", prose, re.I)
    assert not forbidden, f"Forbidden vague phrases: {forbidden}"
    assert not re.search(r"\b\w+['\u2019](?:s|t|re|ve|ll|d|m)\b", prose, re.I)
    acronyms = ["ARI", "BMU", "CQE", "DoS", "kNN", "NMI", "OOF", "QE", "SOM", "TE", "UNSW-NB15"]
    acronym_pattern = r"\b(?:" + "|".join(acronyms) + r")\b"
    for command in ["section", "subsection", "caption"]:
        for argument in command_arguments(sections, command):
            assert not re.search(acronym_pattern, argument), f"Acronym in {command}: {argument}"
    assert "RQ" not in sections and "KMeans" not in sections
    assert "^{(t)}" not in sections and r"\sigma_t" not in sections
    warnings = {
        "listed_pronouns": re.findall(
            r"\b(?:this|these|it|its|them|they|which)\b", prose, re.I
        ),
        "spelled_multidigit_numbers": re.findall(
            r"\b(?:ten|eleven|twelve|twenty|thirty|forty|fifty|hundred)\b", prose, re.I
        ),
        "long_sentences_over_35_words": [
            s.strip()
            for paragraph in re.split(r"\n\s*\n", prose)
            for s in re.split(r"(?<=[.!?])\s+", paragraph)
            if len(re.findall(r"\b[\w-]+\b", s)) > 35
        ],
    }
    labels = re.findall(r"\\label\{([^}]+)\}", sections)
    references = re.findall(r"\\ref\{([^}]+)\}", sections)
    assert len(labels) == len(set(labels))
    displays = {x for x in labels if x.startswith(("fig:", "tab:"))}
    assert displays <= set(references), "Unreferenced figure/table"
    assert set(references) <= set(labels), "Missing label"
    assert len([x for x in labels if x.startswith("fig:")]) == 6
    assert len([x for x in labels if x.startswith("tab:")]) == 4
    citations = {
        key for group in re.findall(r"\\cite\{([^}]+)\}", sections) for key in group.split(",")
    }
    bib = (REPORT / "references.bib").read_text()
    keys = set(re.findall(r"@(?!IEEEtranBSTCTL)\w+\{([^,]+),", bib, re.I))
    assert citations == keys, "Missing or unused bibliography entry"
    bbl = (REPORT / "27284808RW741assignment4.bbl").read_text()
    ordered = re.findall(r"\\bibitem\{([^}]+)\}", bbl)
    author_surnames = {}
    for key in keys:
        entry = re.search(r"@\w+\{" + re.escape(key) + r",(.*?)(?=\n@|\Z)", bib, re.S).group(1)
        first_author = re.search(r"author\s*=\s*\{([^}]+)\}", entry).group(1).split(" and ")[0]
        author_surnames[key] = first_author.split()[-1].lower()
    assert len(ordered) == len(keys) and set(ordered) == keys
    assert [author_surnames[k] for k in ordered] == sorted(author_surnames.values())
    for name in [
        "representation",
        "geometry",
        "sensitivity",
        "fragmentation",
        "transfer",
        "classes",
    ]:
        for suffix in ["pdf", "png"]:
            assert (OUT / f"{name}.{suffix}").stat().st_size > 1000

    pdf = REPORT / "27284808RW741assignment4.pdf"
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"]).decode()
    pages = [p for p in text.split("\f") if p.strip()]
    assert "18 October 2026" in pages[0], "Missing submission date on title page"
    main_pages = next(i for i, page in enumerate(pages) if "appendix a" in page.lower())
    assert 8 <= main_pages <= 10, "Main content exceeds the agreed length"
    assert len(pages) <= 12 and "references" in pages[-1].lower()
    assert "vii. conclusion" in "\n".join(pages[:main_pages]).lower()
    log = (REPORT / "27284808RW741assignment4.log").read_text()
    assert not re.search(r"Overfull|undefined|Fatal error|LaTeX Error", log, re.I)
    fonts = subprocess.check_output(["pdffonts", str(pdf)]).decode()
    assert "Type 3" not in fonts
    objects = subprocess.check_output(["mutool", "show", str(pdf), "grep"], text=True)
    active_links = re.findall(
        r"/Subtype\s*/Link|/S\s*/(?:URI|GoToR|Launch)|/URI\s*[<(]", objects
    )
    assert not active_links, "Active PDF link/action found"
    result = {
        "status": "passed",
        "frozen_report_inputs": len(manifest["inputs"]),
        "pages": len(pages),
        "main_content_pages": main_pages,
        "figures": 6,
        "main_tables": 4,
        "references": len(keys),
        "fits": 0,
        "prediction_computations": 0,
        "pdf_sha256": fingerprint(pdf),
        "underfull_warnings": len(re.findall(r"Underfull", log)),
        "overfull_boxes": 0,
        "unresolved_citations_or_references": 0,
        "numeric_report_csvs_checked": 9,
        "bibliography_alphabetical": True,
        "active_pdf_links_or_external_actions": len(active_links),
        "language_hard_checks": "passed",
        "language_review_warnings": warnings,
    }
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
