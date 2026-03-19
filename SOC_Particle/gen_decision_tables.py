#!/usr/bin/env python3
"""
gen_decision_tables.py
======================
Auto-generate NewDecisionTables.md from DecisionTables.ods using odfpy.

Sheets converted:
  Single Hard Faults, Soft Faults, Vb Selection,
  Ib Active-Standby Selection, Ib Hi-Lo Selection, Annunciation

Usage:
  python3 gen_decision_tables.py [--ods PATH] [--out PATH]

Column conventions preserved from existing DecisionTables.md:
  ·   = don't-care (empty cell in ODS)
  ||  = separator between condition inputs and failure / select results
  T/F = boolean conditions
"""

import argparse
import sys
from pathlib import Path

try:
    from odf.opendocument import load
    from odf.table import Table, TableRow, TableCell
    from odf.text import P
except ImportError:
    sys.exit("odfpy not found — run:  pip3 install odfpy --break-system-packages")

# ── Constants ──────────────────────────────────────────────────────────────────

DOT = '·'   # don't-care placeholder
SHEETS = [
    'Single Hard Faults',
    'Soft Faults',
    'Vb Selection',
    'Ib Active-Standby Selection',
    'Ib Hi-Lo Selection',
    'Annunciation - GUI Plot',
    'Annunciation - Serial',
]

# ── ODS helpers ────────────────────────────────────────────────────────────────

def cell_text(cell) -> str:
    parts = []
    for p in cell.getElementsByType(P):
        for n in p.childNodes:
            parts.append(str(n))
    return ''.join(parts).strip()


def read_sheet_as_dicts(sheet) -> list[dict]:
    """
    Return one dict per row: {col_index: str_value}.
    Handles both table:number-columns-repeated (sparse empty cells) and
    table:number-columns-spanned (merged cells).
    """
    result = []
    for tr in sheet.getElementsByType(TableRow):
        row: dict[int, str] = {}
        col = 0
        for c in tr.getElementsByType(TableCell):
            txt = cell_text(c)
            span   = int(c.getAttribute('numbercolumnsspanned') or 1)
            repeat = int(c.getAttribute('numbercolumnsrepeated') or 1)
            if txt:
                for r in range(repeat):
                    row[col + r * span] = txt
            col += span * repeat
        result.append(row)
    return result


def row_at(row: dict, col: int, default: str = '') -> str:
    return row.get(col, default)


def dict_to_list(row: dict, max_col: int) -> list[str]:
    return [row.get(i, '') for i in range(max_col + 1)]


# ── Header / separator analysis ────────────────────────────────────────────────

def build_header_list(header_dict: dict) -> list[tuple[int, str]]:
    """Return [(col_idx, name), ...] for all named (non-empty) header columns."""
    return sorted((k, v) for k, v in header_dict.items() if k > 0 and v)


def find_output_sep_col(cat_dict: dict, header_dict: dict) -> int | None:
    """
    Find the first column index belonging to the 'output' group.

    Primary: use the category row (row 0).  Look for labels containing
    'result' or 'output' (e.g. 'select result', 'failure result',
    'dispw plot result').  The column of the *first* such label is the
    output separator.

    Fallback: scan named header columns for the last blank-column gap;
    outputs start at the first named column after that gap.
    """
    OUTPUT_KEYWORDS = ('result', 'output', 'dispw')

    for col in sorted(cat_dict):
        if col == 0:
            continue
        label = cat_dict[col].lower()
        if any(kw in label for kw in OUTPUT_KEYWORDS):
            return col

    # Fallback: gap-based heuristic on the header row
    named_cols = sorted(k for k, v in header_dict.items() if k > 0 and v)
    last_gap_end = None
    prev = named_cols[0] if named_cols else None
    for c in named_cols[1:]:
        if c - prev > 1:
            last_gap_end = c
        prev = c
    return last_gap_end


# ── Active-column detection ────────────────────────────────────────────────────

def active_cols_for_section(rows: list[dict], header_dict: dict,
                            out_start: int | None = None) -> list[int]:
    """
    Return only the column indices that have non-empty data in at least one row.
    For the INPUT zone (cols < out_start), also fill in named header columns
    within contiguous clusters of active input columns (gap <= 1 blank col).
    Output zone: only include columns with actual data — no gap-fill.
    """
    active: set[int] = set()
    for row in rows:
        for c, v in row.items():
            if c > 0 and v:
                active.add(c)
    if not active:
        return []

    # Split active cols into input vs output
    if out_start is None:
        inp = sorted(active)
        out: list[int] = []
    else:
        inp = sorted(c for c in active if c < out_start)
        out = sorted(c for c in active if c >= out_start)

    # For inputs: fill in named header cols that sit in small gaps (≤ 1 blank)
    filled_inp: set[int] = set(inp)
    for i in range(len(inp) - 1):
        a, b = inp[i], inp[i + 1]
        if b - a <= 2:                # gap of at most 1 unnamed column
            for c in range(a + 1, b):
                if header_dict.get(c):
                    filled_inp.add(c)

    return sorted(filled_inp | set(out))


def row_is_empty(row: dict) -> bool:
    """True if the row has no data in columns > 0."""
    return not any(c > 0 for c in row)


# ── Header word-wrap ───────────────────────────────────────────────────────────

# Headers longer than this are wrapped onto two lines.
HEADER_WRAP = 8

# Operators tried as split points (operator goes to START of line 2).
# Logical connectives take priority over comparisons; em-dash is last
# because it often appears mid-variable-name rather than as a separator.
_SPLIT_OPS = (' or ', ' and ', ' >= ', ' <= ', ' > ', ' < ', ' – ')


def split_header(text: str) -> tuple[str, str]:
    """
    Split a long header label into (line1, line2).
    Returns (text, '') when no split is needed or possible.

    Priority:
      1. Comparison / logical operator: keep operator at start of line 2.
      2. Parenthetical suffix ' (…)': parenthesis on line 2.
      3. Fused operator with no spaces: '>=', '<='.
      4. Space nearest the midpoint.
    """
    if len(text) <= HEADER_WRAP:
        return text, ''

    # 1. Spaced operators
    for pat in _SPLIT_OPS:
        idx = text.find(pat)
        if 0 < idx < len(text) - len(pat):
            return text[:idx], pat.lstrip() + text[idx + len(pat):]

    # 2. Parenthetical  'foo (bar)' → ('foo', '(bar)')
    idx = text.find(' (')
    if 0 < idx:
        return text[:idx], text[idx + 1:]

    # 3. Fused operators
    for pat in ('>=', '<='):
        idx = text.find(pat)
        if 0 < idx < len(text) - 2:
            return text[:idx], text[idx:]

    # 4. Nearest space to midpoint
    mid = len(text) // 2
    spaces = [i for i, ch in enumerate(text) if ch == ' ']
    if spaces:
        best = min(spaces, key=lambda i: abs(i - mid))
        return text[:best], text[best + 1:]

    return text, ''


# ── Markdown table renderer ────────────────────────────────────────────────────

def render_table(name: str, header_dict: dict, section_rows: list[dict],
                 out_start: int | None) -> str:
    """
    Render a named sub-table as a markdown fenced code block.
    Long column headers are wrapped onto two lines (see split_header).
    Rows that are entirely empty (no data in cols > 0) are skipped.
    """
    # Drop all-empty data rows
    data_rows = [r for r in section_rows if not row_is_empty(r)]
    if not data_rows:
        return ''

    act = active_cols_for_section(data_rows, header_dict, out_start)
    if not act:
        return ''

    # Determine output separator position within `act`
    sep_pos: int | None = None
    if out_start is not None:
        for j, c in enumerate(act):
            if c >= out_start:
                sep_pos = j
                break

    # Collect header labels and compute two-line splits
    h_labels = [header_dict.get(c, '') for c in act]
    h1: list[str] = []
    h2: list[str] = []
    for lbl in h_labels:
        l1, l2 = split_header(lbl)
        h1.append(l1)
        h2.append(l2)
    two_line = any(h2)

    # Compute per-column widths (both header lines + data)
    widths: dict[int, int] = {}
    for i, c in enumerate(act):
        w = max(len(h1[i]), len(h2[i]))
        for row in data_rows:
            w = max(w, len(row.get(c, DOT)))
        widths[c] = max(w, 1)

    row_num_w = max(1, len(str(len(data_rows))))

    def cell(val: str, col: int) -> str:
        return val.ljust(widths[col])

    def mk_row(cells_left: list[str], cells_right: list[str] | None) -> str:
        left = ' | '.join(cells_left)
        if cells_right is not None:
            return f'| {left} || {" | ".join(cells_right)} |'
        return f'| {left} |'

    def io_split(vals: list[str]) -> tuple[list[str], list[str] | None]:
        """Split vals list at the input/output boundary."""
        if sep_pos is None:
            return vals, None
        # vals[0] = '#'; vals[1:] map to act[0], act[1], ...
        # sep_pos is index in act, so offset by 1 in vals
        return vals[:sep_pos + 1], vals[sep_pos + 1:]

    # Build rows
    h_row1 = ['#'.ljust(row_num_w)] + [cell(h1[i], act[i]) for i in range(len(act))]
    h_row2 = [' ' * row_num_w]      + [cell(h2[i], act[i]) for i in range(len(act))]
    s_row  = ['-' * row_num_w]      + ['-' * widths[c]      for c in act]

    lines = [f'### {name}', '', '```text']
    lines.append(mk_row(*io_split(h_row1)))
    if two_line:
        lines.append(mk_row(*io_split(h_row2)))
    lines.append(mk_row(*io_split(s_row)))

    for rnum, row in enumerate(data_rows, 1):
        d_row = [str(rnum).ljust(row_num_w)]
        for c in act:
            v = row.get(c, '')
            d_row.append(cell(v if v else DOT, c))
        lines.append(mk_row(*io_split(d_row)))

    lines.append('```')
    return '\n'.join(lines)


# ── Section / note parsing ─────────────────────────────────────────────────────

def is_new_section(col0: str) -> bool:
    """
    True if col0 starts a new named sub-table (e.g. 'Fault::tb_check').
    Sub-labels like '- first row Tb' or note lines are NOT new sections.
    """
    if not col0:
        return False
    if col0.startswith('-'):   # Annunciation sub-labels
        return False
    if col0.lower().startswith('note'):
        return False
    return True


def parse_sections(all_rows: list[dict]) -> list[tuple[str, list[dict], list[str]]]:
    """
    Split sheet rows (starting at row index 2) into sections.
    Returns list of (section_name, data_rows, note_lines).
    """
    sections: list[tuple[str, list[dict], list[str]]] = []
    current_name: str | None = None
    current_rows: list[dict] = []
    notes: list[str] = []
    in_notes = False

    for row in all_rows[2:]:
        col0 = row.get(0, '').strip()

        if col0.lower().startswith('note'):
            in_notes = True
            continue                      # skip the 'Notes:' header itself

        if in_notes:
            # A truly empty dict (no columns at all) ends the notes block.
            # row_is_empty() only checks cols > 0, so note lines that have
            # content only in col 0 would falsely appear empty to it.
            if not row:
                in_notes = False          # blank row ends the notes block
                # fall through — empty row is not a section; safe to skip
            else:
                # Note lines: text is in col 0; fall back to other cols.
                txt = col0 or ' '.join(
                    v for k, v in sorted(row.items()) if k > 0 and v
                )
                if txt:
                    notes.append(txt)
                continue

        if is_new_section(col0):
            if current_name is not None:
                sections.append((current_name, current_rows, notes))
                notes = []
            current_name = col0
            current_rows = [row]
        else:
            if current_name is not None:
                current_rows.append(row)

    if current_name and current_rows:
        sections.append((current_name, current_rows, notes))

    return sections


# ── Sheet processor ────────────────────────────────────────────────────────────

def process_sheet(doc, sheet_name: str) -> str:
    sheet_obj = None
    for s in doc.spreadsheet.getElementsByType(Table):
        if s.getAttribute('name') == sheet_name:
            sheet_obj = s
            break
    if sheet_obj is None:
        return f'## {sheet_name}\n\n*Sheet not found*\n'

    all_rows = read_sheet_as_dicts(sheet_obj)
    if len(all_rows) < 2:
        return f'## {sheet_name}\n\n*Empty*\n'

    cat_dict    = all_rows[0]               # row index 0 = category labels
    header_dict = all_rows[1]               # row index 1 = column headers
    out_start = find_output_sep_col(cat_dict, header_dict)

    sections = parse_sections(all_rows)

    parts = [f'## {sheet_name}', '']
    for sec_name, sec_rows, sec_notes in sections:
        blk = render_table(sec_name, header_dict, sec_rows, out_start)
        if blk:
            parts.append(blk)
            if sec_notes:
                parts.append('')
                parts.append('> **Notes:**')
                for n in sec_notes:
                    parts.append('>')
                    parts.append(f'> {n}')
            parts.append('')

    return '\n'.join(parts)


# ── Document skeleton ──────────────────────────────────────────────────────────

SUMMARY = """\
## Summary

Auto-generated from `DecisionTables.ods`. The Active-Standby sheet is kept for
reference; Hi-Lo is the active strategy.

All parameters are unlatched unless `latch` specifically appears in the name.
Hard Faults take precedence over Soft Faults. Soft Faults are used in reasoning.
`·` = don't-care / any value.  `||` separates condition inputs from results.

**Abbreviations**

| Abbrev | Meaning |
| ------ | ------- |
| Rf     | `reset_all_faults` |
| Ff     | `ap.fake_fault` |
| si     | `sp.ib_force` |
| ibl    | last good ib_sel value |
| vbl    | last good vb_sel value |
"""

TOC = {
    'Single Hard Faults':          'single-hard-faults',
    'Soft Faults':                 'soft-faults',
    'Vb Selection':                'vb-selection',
    'Ib Active-Standby Selection': 'ib-active-standby-selection',
    'Ib Hi-Lo Selection':          'ib-hi-lo-selection',
    'Annunciation':                'annunciation',
}


def build_toc() -> str:
    lines = ['## Contents', '']
    for name, anchor in TOC.items():
        lines.append(f'- [{name}](#{anchor})')
    return '\n'.join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ods', default='DecisionTables.ods')
    ap.add_argument('--out', default='DecisionTables.md')
    args = ap.parse_args()

    ods_path = Path(args.ods)
    out_path = Path(args.out)

    if not ods_path.exists():
        sys.exit(f'ODS file not found: {ods_path}')

    print(f'Loading {ods_path} …')
    doc = load(str(ods_path))

    sheet_blocks: list[str] = []
    for name in SHEETS:
        print(f'  {name}')
        sheet_blocks.append(process_sheet(doc, name))

    divider = '\n\n---\n\n'
    output = '\n'.join([
        '<style>pre { white-space: pre; overflow-x: auto; }</style>',
        '',
        '# Decision Tables — SOC_Particle',
        '',
        build_toc(),
        '',
        '---',
        '',
        SUMMARY,
        '',
        '---',
        '',
        divider.join(sheet_blocks),
    ])

    out_path.write_text(output, encoding='utf-8')
    print(f'Written → {out_path}')


if __name__ == '__main__':
    main()
