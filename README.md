# HyperMOA

HyperMOA is a multi-omics spatial slice alignment framework with adaptive
hypergraph learning and batch-centering correction for cross-slice representation
learning, spatial registration, and 3D tissue reconstruction.

Core features:

- Multi-omics multi-slice alignment for RNA with ADT or epigenomic features.
- Hypergraph-enhanced graph attention with adaptive hyperedges.
- Batch-centering correction to reduce slice-specific representation shifts.
- Slice ordering, location alignment, 3D reconstruction, and evaluation exports.

Data are expected in a shared directory next to the repository:

```bash
/root/autodl-tmp/data
```

You can override that location with:

```bash
export HYPERMOA_DATA_DIR=/path/to/data
```
