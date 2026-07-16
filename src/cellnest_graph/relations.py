"""Relation-type registry: turn an LR table into distinct integer relation ids.

Mirrors CellNEST's ``l_r_pair`` / ``ligand_dict_dataset`` / ``cell_cell_contact`` logic
(``docs/cellnest_graph_reference.md`` §2), but as a small, testable class. A relation type
corresponds to one (ligand, receptor) pair; each gets a stable integer id.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Annotation strings that mark a receptor as contact-restricted (juxtacrine). CellNEST uses
# the exact string "Cell-Cell Contact"; we also accept common variants case-insensitively.
_CONTACT_MARKERS = ("cell-cell contact", "cell cell contact", "juxtacrine", "contact")


def _is_contact(annotation: str) -> bool:
    a = str(annotation).strip().lower()
    return any(marker in a for marker in _CONTACT_MARKERS)


@dataclass
class RelationRegistry:
    """Maps (ligand, receptor) -> relation id and tracks contact-restricted receptors.

    Attributes
    ----------
    pair_to_id : dict[tuple[str, str], int]
        (ligand, receptor) -> relation id.
    ligand_to_receptors : dict[str, list[str]]
        ligand -> receptors it pairs with (present in the dataset).
    contact_receptors : set[str]
        receptors whose pair is annotated as cell-cell contact.
    table : pandas.DataFrame
        relation_id, ligand, receptor, is_contact.
    """

    pair_to_id: dict[tuple[str, str], int]
    ligand_to_receptors: dict[str, list[str]]
    contact_receptors: set[str]
    table: pd.DataFrame

    @classmethod
    def from_lr_table(
        cls,
        lr_pairs: pd.DataFrame,
        present_genes: set[str],
        extra_contact_receptors: set[str] | None = None,
    ) -> "RelationRegistry":
        """Build a registry, keeping only pairs whose *both* genes are in ``present_genes``.

        Relation ids are assigned in first-seen order over the (filtered) input rows, giving
        deterministic, dataset-specific ids just like CellNEST.
        """
        required = {"ligand", "receptor"}
        if not required.issubset(lr_pairs.columns):
            raise ValueError(
                f"lr_pairs must have columns {required}, got {list(lr_pairs.columns)}"
            )
        has_annotation = "annotation" in lr_pairs.columns

        pair_to_id: dict[tuple[str, str], int] = {}
        ligand_to_receptors: dict[str, list[str]] = {}
        contact_receptors: set[str] = set(extra_contact_receptors or set())
        rows = []
        next_id = 0
        for r in lr_pairs.itertuples(index=False):
            ligand = str(r.ligand).upper()
            receptor = str(r.receptor).upper()
            if ligand not in present_genes or receptor not in present_genes:
                continue
            key = (ligand, receptor)
            is_contact = (
                _is_contact(getattr(r, "annotation", "")) if has_annotation else False
            )
            if is_contact:
                contact_receptors.add(receptor)
            if key in pair_to_id:
                continue
            pair_to_id[key] = next_id
            ligand_to_receptors.setdefault(ligand, [])
            if receptor not in ligand_to_receptors[ligand]:
                ligand_to_receptors[ligand].append(receptor)
            rows.append(
                {
                    "relation_id": next_id,
                    "ligand": ligand,
                    "receptor": receptor,
                    "is_contact": is_contact,
                }
            )
            next_id += 1

        table = pd.DataFrame(
            rows, columns=["relation_id", "ligand", "receptor", "is_contact"]
        )
        return cls(pair_to_id, ligand_to_receptors, contact_receptors, table)

    def __len__(self) -> int:
        return len(self.pair_to_id)

    @property
    def ligands(self) -> list[str]:
        return list(self.ligand_to_receptors.keys())

    @property
    def genes(self) -> set[str]:
        """All ligand and receptor genes referenced by kept relations."""
        g = set(self.ligand_to_receptors.keys())
        for recs in self.ligand_to_receptors.values():
            g.update(recs)
        return g
